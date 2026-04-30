"""
train_quic.py — Train GNB on QUIC traces (dual-profile, 6D features)
Paper reference: Nebby §3.4 step 4

HOW THIS RELATES TO train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train.py uses extract_features_dual_profile() which internally calls
compute_bif() from bif.py — the TCP-aware BiF estimator.

This file is structurally identical to train.py.  The only change is
that build_dataset_quic() calls _get_quic_features_dual() which uses
compute_bif_quic() from quic_bif.py instead of compute_bif().

SHARED MODEL VS SEPARATE MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
By default this writes to ../models/gnb_quic.pkl — a separate model
from the TCP gnb.pkl.  This is recommended because:

  1. QUIC stacks (quiche, mvfst, xquic) implement CUBIC / BBR differently
     from Linux kernel TCP.  Their polynomial coefficient distributions
     have slightly different μ and σ than the TCP versions (paper §4.3).

  2. Keeping separate models lets you compare TCP vs QUIC accuracy and
     retrain each independently as more traces accumulate.

  3. classify_quic.py will automatically use gnb_quic.pkl if it exists,
     falling back to gnb.pkl (the TCP model) otherwise.

If you prefer a SINGLE shared model for both TCP and QUIC, set:
    MODEL_DIR = '../models'
    MODEL_STEM = 'gnb'          ← same as TCP
This works because the polynomial shape is protocol-agnostic; the CCA's
growth law is the same regardless of whether it runs over TCP or QUIC.

FILE NAMING CONVENTION
━━━━━━━━━━━━━━━━━━━━━━
QUIC CSVs produced by pcap2csv_quic.sh should be named:
    cc-<cca>_quic_<delay>ms_<timestamp>.csv
    e.g.  cc-cubic_quic_50ms_20240301T1200.csv
          cc-cubic_quic_100ms_20240301T1201.csv
          cc-bbr_quic_50ms_20240301T1210.csv

pair_traces_quic() groups files by CCA and pairs consecutive
(50ms, 100ms) files, same logic as pair_traces() in train.py.

Usage:
    python3 train_quic.py

Outputs (saved to ../models/):
    gnb_quic.pkl
    label_encoder_quic.pkl
    training_report_quic.txt
    feature_clusters_2d_quic.png
"""

import os, sys, glob, re
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics         import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

# ── your existing modules (unchanged) ────────────────────────────────────────
from bif        import smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features

# ── new QUIC-only module ──────────────────────────────────────────────────────
from quic_bif   import compute_bif_quic

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR      = '../candidates-measurements-quic'   # directory of *_quic.csv files
MODEL_DIR    = '../models'
MODEL_STEM   = 'gnb_quic'                          # → gnb_quic.pkl
SERVER_IP    = None          # None = auto-detect from CSV
MIN_SEGMENTS = 5
CV_FOLDS     = 5

RATE_BASED_CCAS = {'bbr', 'bbr2', 'bbr3'}

_PALETTE = [
    '#2196F3', '#e63946', '#FF9800', '#9C27B0', '#00BCD4',
    '#4CAF50', '#795548', '#607D8B', '#E91E63', '#FF5722',
    '#009688', '#FFC107',
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PAIR TRACES  (mirrors pair_traces() in train.py)
# ══════════════════════════════════════════════════════════════════════════════

def pair_traces_quic(csv_dir):
    """
    Group QUIC CSV files by CCA and pair consecutive (50ms, 100ms) files.

    Filename pattern:  cc-<cca>_quic_<anything>.csv
    Consecutive files are paired as (50ms run, 100ms run) in sorted order,
    matching the order pcap2csv_quic.sh produces them.

    Returns
    -------
    pairs    : list of (label, path_50ms, path_100ms)
    unpaired : list of filenames that could not be paired
    """
    # Match both *_quic_*.csv and *_tcp.csv patterns so this works whether
    # the user names files with _quic_ or keeps _tcp suffix on QUIC csvs.
    patterns = [
        os.path.join(csv_dir, '*_quic_*.csv'),
        os.path.join(csv_dir, '*_quic*.csv'),
    ]
    files = sorted(set(
        f for pat in patterns for f in glob.glob(pat)
    ))

    if not files:
        # Fallback: any CSV that has cc-<cca> pattern
        files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))

    by_cca = defaultdict(list)
    for f in files:
        cc = re.search(r'cc-(\w+)[_.]', os.path.basename(f))
        if cc:
            by_cca[cc.group(1)].append(f)

    pairs    = []
    unpaired = []

    for cca, flist in sorted(by_cca.items()):
        if cca in RATE_BASED_CCAS:
            continue   # BBR handled by rule detector

        for i in range(0, len(flist) - 1, 2):
            pairs.append((cca, flist[i], flist[i + 1]))

        if len(flist) % 2 != 0:
            unpaired.append(os.path.basename(flist[-1]))

    return pairs, unpaired


# ══════════════════════════════════════════════════════════════════════════════
# 2.  QUIC FEATURE EXTRACTION  (mirrors _get_features_from_csv in features.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_quic_features(csv_path, rtt_s, server_ip=None):
    """
    Full QUIC preprocessing pipeline for one CSV.
    Returns numpy array of shape (n_segments, 3).
    Identical to features._get_features_from_csv() but uses compute_bif_quic.
    """
    t, bif         = compute_bif_quic(csv_path, server_ip)   # ← only difference
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
    segments       = segment_bif(t_ss, bif_ss)
    return extract_features(segments)   # (n_segs, 3)


def _get_quic_features_dual(csv_50ms, csv_100ms, server_ip=None):
    """
    Extract 6D features from a QUIC dual-profile pair.
    Returns (feats_6d, n_paired_segments).
    Mirrors extract_features_dual_profile() in features.py.
    """
    f50  = _get_quic_features(csv_50ms,  rtt_s=0.10, server_ip=server_ip)
    f100 = _get_quic_features(csv_100ms, rtt_s=0.20, server_ip=server_ip)

    n = min(len(f50), len(f100))
    if n == 0:
        return np.empty((0, 6)), 0

    feats_6d = np.hstack([f50[:n], f100[:n]])   # (n, 6)
    return feats_6d, n


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD DATASET  (mirrors build_dataset() in train.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_quic(csv_dir, server_ip=SERVER_IP):
    """
    Process every paired QUIC (50ms, 100ms) CSV set.
    Returns X of shape (n_segments, 6) and y of shape (n_segments,).
    """
    pairs, unpaired = pair_traces_quic(csv_dir)

    if not pairs:
        raise ValueError(
            f"No paired QUIC CSV files found in {csv_dir}.\n"
            "Expected pattern:  cc-<cca>_quic_<delay>ms_<ts>.csv\n"
            "Run pcap2csv_quic.sh on your QUIC pcap captures first."
        )

    if unpaired:
        print(f"  WARNING: {len(unpaired)} unpaired file(s) — skipped:")
        for f in unpaired:
            print(f"    {f}")

    X, y          = [], []
    skipped_error = []
    skipped_noseg = []
    per_class     = {}

    print(f"\n  Found {len(pairs)} QUIC trace pair(s)\n")
    print(f"  {'CCA':<12} {'50ms FILE':<45} {'100ms FILE':<45} {'SEGS':>5}")
    print(f"  {'─'*12} {'─'*45} {'─'*45} {'─'*5}")

    for label, f50, f100 in pairs:
        name50  = os.path.basename(f50)
        name100 = os.path.basename(f100)

        try:
            feats, n = _get_quic_features_dual(f50, f100, server_ip)
        except Exception as e:
            print(f"  {label:<12} {name50:<45} ERROR: {e}")
            skipped_error.append((name50, name100, str(e)))
            continue

        if n == 0:
            print(f"  {label:<12} {name50:<45} {name100:<45} {'0':>5}  NO SEGS")
            skipped_noseg.append((name50, name100))
            continue

        print(f"  {label:<12} {name50:<45} {name100:<45} {n:>5}")
        per_class[label] = per_class.get(label, 0) + n

        for fv in feats:
            X.append(fv)
            y.append(label)

    return (np.array(X), np.array(y),
            skipped_error, skipped_noseg, per_class)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VALIDATE  (identical to train.py)
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_filter(X, y, min_segments=MIN_SEGMENTS):
    classes, counts = np.unique(y, return_counts=True)
    removed = []
    for cls, cnt in zip(classes, counts):
        if cnt < min_segments:
            print(f"  WARNING: '{cls}' has only {cnt} segment(s) "
                  f"(need ≥ {min_segments}) — removing")
            removed.append(cls)
    if removed:
        mask = np.isin(y, removed, invert=True)
        X, y = X[mask], y[mask]
    return X, y, removed


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FEATURE CLUSTER PLOTS  (mirrors train.py, relabelled for QUIC)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_clusters_2d_quic(X, y, out_dir):
    classes = sorted(set(y))
    cmap    = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(classes)}

    dim_labels = ['a₁(cubic)', 'b₁(quad)', 'c₁(lin)',
                  'a₂(cubic)', 'b₂(quad)', 'c₂(lin)']
    proj_pairs = [
        (0, 1, '50ms QUIC: a₁ vs b₁'),
        (3, 4, '100ms QUIC: a₂ vs b₂'),
        (1, 2, '50ms QUIC: b₁ vs c₁'),
        (4, 5, '100ms QUIC: b₂ vs c₂'),
        (0, 3, 'Cross-profile: a₁ vs a₂'),
        (1, 4, 'Cross-profile: b₁ vs b₂'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for ax, (xi, yi, title) in zip(axes, proj_pairs):
        for cls in classes:
            mask = y == cls
            ax.scatter(X[mask, xi], X[mask, yi],
                       c=cmap[cls], label=cls,
                       s=20, alpha=0.65, edgecolors='none')
        ax.set_xlabel(dim_labels[xi], fontsize=9)
        ax.set_ylabel(dim_labels[yi], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=7, loc='best',
                   ncol=2 if len(classes) > 6 else 1)
    fig.suptitle(
        'QUIC Feature Clusters — 6D Dual-Profile Space\n'
        'Subscript 1 = 50ms profile, 2 = 100ms profile',
        fontsize=13,
    )
    plt.tight_layout()

    path = os.path.join(out_dir, 'feature_clusters_2d_quic.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAIN GNB  (identical logic to train.py)
# ══════════════════════════════════════════════════════════════════════════════

def train_gnb_quic(X, y, cv_folds=CV_FOLDS):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n  Classes : {list(le.classes_)}")
    print(f"  Segments: {len(X)}  |  Features: {X.shape[1]}D")

    min_count    = int(np.bincount(y_enc).min())
    actual_folds = min(cv_folds, min_count)
    cv_report    = None

    if actual_folds < 2:
        print(f"\n  WARNING: smallest class has {min_count} segment(s). "
              "Cannot run CV — fitting on full data.")
        gnb = GaussianNB()
        gnb.fit(X, y_enc)
    else:
        if actual_folds < cv_folds:
            print(f"\n  Reducing CV folds: {cv_folds} → {actual_folds}")

        skf  = StratifiedKFold(n_splits=actual_folds,
                                shuffle=True, random_state=42)
        y_cv = cross_val_predict(GaussianNB(), X, y_enc, cv=skf)
        acc  = accuracy_score(y_enc, y_cv)

        print(f"\n  {actual_folds}-fold CV accuracy: {acc:.1%}\n")
        cv_report = classification_report(
            y_enc, y_cv,
            target_names=le.classes_,
            digits=3, zero_division=0,
        )
        print(cv_report)

        gnb = GaussianNB()
        gnb.fit(X, y_enc)

    return gnb, le, cv_report


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_report_quic(per_class, removed, skipped_error,
                     skipped_noseg, le, cv_report, out_dir):
    lines = [
        "Nebby QUIC Training Report — Dual Profile (6D Features)",
        "=" * 60,
        "",
        "Feature vector: [a1,b1,c1, a2,b2,c2]",
        "  Subscript 1 = 50ms QUIC delay profile",
        "  Subscript 2 = 100ms QUIC delay profile",
        "  BiF estimated via compute_bif_quic() (quic_bif.py)",
        "",
        "CCAs in QUIC GNB model:",
    ]
    for cls in sorted(le.classes_):
        lines.append(f"  {cls:<15}: {per_class.get(cls, 0)} segments")

    if removed:
        lines += ["", "Removed (too few segments):"]
        lines += [f"  {c}" for c in removed]

    if skipped_error:
        lines += ["", "Pairs skipped due to errors:"]
        lines += [f"  {f50} + {f100}: {e}" for f50, f100, e in skipped_error]

    if skipped_noseg:
        lines += ["", "Pairs with no usable segments:"]
        lines += [f"  {f50} + {f100}" for f50, f100 in skipped_noseg]

    if cv_report:
        lines += ["", "Cross-validation report:", cv_report]

    path = os.path.join(out_dir, 'training_report_quic.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_quic(csv_dir=CSV_DIR, model_dir=MODEL_DIR, server_ip=SERVER_IP):
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("  STEP 1 — Building QUIC dual-profile dataset (6D features)")
    print(f"  CSV dir   : {csv_dir}")
    print(f"  Server IP : {server_ip or 'auto-detect'}")
    print(f"  BBR excl. : {RATE_BASED_CCAS}  (rule-based in classify_quic.py)")
    print("=" * 60)

    X, y, skipped_error, skipped_noseg, per_class = build_dataset_quic(
        csv_dir, server_ip
    )

    if len(X) == 0:
        print("\nNo QUIC data collected. Check CSV_DIR and pcap2csv_quic.sh output.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  STEP 2 — Validating dataset")
    print("=" * 60 + "\n")
    X, y, removed = validate_and_filter(X, y)

    print(f"\n  Segment counts per QUIC class:")
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        bar = '█' * min(cnt, 50)
        print(f"  {cls:<15}: {cnt:>4}  {bar}")

    print("\n" + "=" * 60)
    print("  STEP 3 — Plotting 6D QUIC feature clusters")
    print("=" * 60 + "\n")
    plot_feature_clusters_2d_quic(X, y, model_dir)

    print("\n" + "=" * 60)
    print("  STEP 4 — Training Gaussian Naive Bayes on QUIC features")
    print("=" * 60)
    gnb, le, cv_report = train_gnb_quic(X, y)

    print("\n" + "=" * 60)
    print("  STEP 5 — Saving QUIC model")
    print("=" * 60 + "\n")

    gnb_path = os.path.join(model_dir, f'{MODEL_STEM}.pkl')
    le_path  = os.path.join(model_dir, f'label_encoder_{MODEL_STEM.split("_",1)[-1]}.pkl')
    joblib.dump(gnb, gnb_path)
    joblib.dump(le,  le_path)
    print(f"  Saved: {gnb_path}")
    print(f"  Saved: {le_path}")

    save_report_quic(per_class, removed, skipped_error,
                     skipped_noseg, le, cv_report, model_dir)

    print(f"\n  QUIC GNB classes : {list(le.classes_)}")
    print(f"  Feature dim      : {X.shape[1]}D  (dual profile)")
    print(f"  BBR handled      : rule-based detector in classify_quic.py\n")
    return gnb, le


if __name__ == '__main__':
    train_quic()