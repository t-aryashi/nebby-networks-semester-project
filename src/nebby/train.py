"""
train.py — Build dataset and train Gaussian Naive Bayes classifier
Paper reference: Nebby §3.4 step 4 (Clustering and Classification)

IMPORTANT — two-classifier architecture for all 17 CCAs:
  ┌─────────────────────────────────────────────────────┐
  │  BBR  →  rule-based detector in classify.py         │
  │          (NOT trained here — GNB cannot fit BBR)    │
  │                                                     │
  │  All other CCAs  →  GNB on [a, b, c] polynomial    │
  │                      coefficients (trained here)    │
  └─────────────────────────────────────────────────────┘

BBR is rate-based. Its BiF does NOT produce a sawtooth — polynomial
fitting gives meaningless coefficients that corrupt the GNB clusters.

Usage:
    python3 train.py

Outputs:
    ../models/gnb.pkl
    ../models/label_encoder.pkl
    ../models/training_report.txt
    ../models/feature_clusters.png     3D scatter — replicates paper Figure 7
    ../models/feature_clusters_2d.png  three 2D projections
"""

import os, sys, glob, re
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics         import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR      = '../candidates-measurements'
MODEL_DIR    = '../models'
SERVER_IP    = '10.0.0.1'
RTT_S        = 0.1          # seconds  (2 × 50 ms one-way delay)
MIN_SEGMENTS = 5            # drop any class with fewer segments than this
CV_FOLDS     = 5            # stratified k-fold for accuracy estimate

# CCAs handled by the RULE-BASED detector — excluded from GNB training
RATE_BASED_CCAS = {'bbr', 'bbr2', 'bbr3'}

# Full expected set — used for warnings when a CCA has no data yet
ALL_CCAS = {
    'bbr', 'bic', 'cdg', 'cubic', 'dctcp', 'highspeed',
    'htcp', 'hybla', 'illinois', 'lp', 'nv', 'reno',
    'scalable', 'vegas', 'veno', 'westwood', 'yeah',
}

# Colour palette for feature-cluster plots (17 distinct colours)
_PALETTE = [
    '#e63946', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4',
    '#4CAF50', '#795548', '#607D8B', '#E91E63', '#FF5722',
    '#009688', '#FFC107', '#3F51B5', '#8BC34A', '#F44336',
    '#673AB7', '#FFEB3B',
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(csv_dir, server_ip=SERVER_IP, rtt_s=RTT_S):
    """
    Process every *_tcp.csv, skip BBR (rule-based), return (X, y).

    X : (n_segments, 3)  — polynomial coefficients [a, b, c]
    y : (n_segments,)    — CCA label string
    """
    X, y          = [], []
    skipped_bbr   = []
    skipped_error = []
    skipped_noseg = []
    per_class     = {}

    files = sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv')))
    if not files:
        raise FileNotFoundError(
            f"No *_tcp.csv files found in {csv_dir}\n"
            "Run generate_dataset.sh first."
        )

    print(f"  Found {len(files)} CSV files\n")
    print(f"  {'FILE':<52} {'LABEL':<12} {'SEGS':>5}  STATUS")
    print(f"  {'─'*52} {'─'*12} {'─'*5}  {'─'*12}")

    for fpath in files:
        fname    = os.path.basename(fpath)
        cc_match = re.search(r'cc-(\w+)_', fname)
        if not cc_match:
            print(f"  {fname:<52} {'?':<12} {'─':>5}  SKIP (no label)")
            continue

        label = cc_match.group(1)

        # ── BBR: skip from GNB, track separately ─────────────────────────
        if label in RATE_BASED_CCAS:
            print(f"  {fname:<52} {label:<12} {'─':>5}  RATE-BASED → rule detector")
            skipped_bbr.append(fpath)
            continue

        # ── Loss-based: full preprocessing + feature extraction ───────────
        try:
            t, bif         = compute_bif(fpath, server_ip)
            t_s, bif_s     = smooth_bif(t, bif, rtt_s)
            t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
            segments       = segment_bif(t_ss, bif_ss)
            feats          = extract_features(segments)
        except Exception as e:
            print(f"  {fname:<52} {label:<12} {'─':>5}  ERROR: {e}")
            skipped_error.append((fname, str(e)))
            continue

        if len(feats) == 0:
            print(f"  {fname:<52} {label:<12} {'0':>5}  NO SEGMENTS")
            skipped_noseg.append(fname)
            continue

        print(f"  {fname:<52} {label:<12} {len(feats):>5}  OK")
        per_class[label] = per_class.get(label, 0) + len(feats)
        for fv in feats:
            X.append(fv)
            y.append(label)

    return (np.array(X), np.array(y),
            skipped_bbr, skipped_error, skipped_noseg, per_class)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VALIDATE — remove classes with too few segments
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_filter(X, y, min_segments=MIN_SEGMENTS):
    """
    Remove classes that have fewer than min_segments.
    Tiny classes crash stratified CV and produce unreliable GNB Gaussians.
    """
    classes, counts = np.unique(y, return_counts=True)
    removed = []

    for cls, cnt in zip(classes, counts):
        if cnt < min_segments:
            print(f"  WARNING: '{cls}' has only {cnt} segment(s) "
                  f"(need ≥ {min_segments}) — removing from GNB training")
            removed.append(cls)

    if removed:
        mask = np.isin(y, removed, invert=True)
        X, y = X[mask], y[mask]

    return X, y, removed


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE CLUSTER PLOTS  (replicates paper Figure 7)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_clusters_3d(X, y, out_dir):
    """3D scatter of [a, b, c] — mirrors paper Figure 7."""
    classes = sorted(set(y))
    cmap    = {cls: _PALETTE[i % len(_PALETTE)]
               for i, cls in enumerate(classes)}

    fig = plt.figure(figsize=(13, 8))
    ax  = fig.add_subplot(111, projection='3d')

    for cls in classes:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=cmap[cls], label=cls, s=20, alpha=0.6,
                   edgecolors='none')

    ax.set_xlabel('a  (cubic coeff)',     fontsize=9)
    ax.set_ylabel('b  (quadratic coeff)', fontsize=9)
    ax.set_zlabel('c  (linear coeff)',    fontsize=9)
    ax.set_title('Polynomial Feature Clusters — All Loss-Based CCAs\n'
                 '(replicates Figure 7 from Nebby paper)', fontsize=11)
    ax.legend(fontsize=7, loc='upper left',
              bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.tight_layout()

    path = os.path.join(out_dir, 'feature_clusters.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_clusters_2d(X, y, out_dir):
    """Three 2D projections of the feature space — easier to read than 3D."""
    classes = sorted(set(y))
    cmap    = {cls: _PALETTE[i % len(_PALETTE)]
               for i, cls in enumerate(classes)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1, 'a (cubic)', 'b (quadratic)'),
             (1, 2, 'b (quadratic)', 'c (linear)'),
             (0, 2, 'a (cubic)', 'c (linear)')]

    for ax, (xi, yi, xl, yl) in zip(axes, pairs):
        for cls in classes:
            mask = y == cls
            ax.scatter(X[mask, xi], X[mask, yi],
                       c=cmap[cls], label=cls,
                       s=18, alpha=0.65, edgecolors='none')
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f'{xl} vs {yl}', fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=7, loc='best',
                   ncol=2 if len(classes) > 8 else 1)
    fig.suptitle('Polynomial Coefficient Clusters — 2D Projections', fontsize=13)
    plt.tight_layout()

    path = os.path.join(out_dir, 'feature_clusters_2d.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN GNB WITH CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def train_gnb(X, y, cv_folds=CV_FOLDS):
    """
    Train GNB, estimate accuracy via stratified k-fold CV,
    then refit on the full dataset for deployment.
    """
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n  Classes in GNB : {list(le.classes_)}")
    print(f"  Total segments : {len(X)}")

    # Determine safe number of folds (can't exceed smallest class count)
    min_class_count = int(np.bincount(y_enc).min())
    actual_folds    = min(cv_folds, min_class_count)

    cv_report = None

    if actual_folds < 2:
        print(f"\n  WARNING: Smallest class has only {min_class_count} segment(s).")
        print("  Cannot run cross-validation. Fitting on full data.\n")
        gnb = GaussianNB()
        gnb.fit(X, y_enc)
    else:
        if actual_folds < cv_folds:
            print(f"\n  Reducing CV folds: {cv_folds} → {actual_folds} "
                  f"(smallest class has {min_class_count} segments)")

        skf  = StratifiedKFold(n_splits=actual_folds,
                                shuffle=True, random_state=42)
        y_cv = cross_val_predict(GaussianNB(), X, y_enc, cv=skf)
        acc  = accuracy_score(y_enc, y_cv)

        print(f"\n  {actual_folds}-fold CV accuracy: {acc:.1%}\n")
        cv_report = classification_report(
            y_enc, y_cv,
            target_names=le.classes_,
            digits=3,
            zero_division=0,
        )
        print(cv_report)

        # Final model on ALL data
        gnb = GaussianNB()
        gnb.fit(X, y_enc)

    return gnb, le, cv_report


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE TRAINING REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_report(per_class, removed, skipped_error,
                skipped_noseg, skipped_bbr, le, cv_report, out_dir):
    lines = [
        "Nebby Training Report",
        "=" * 60,
        "",
        "CCAs handled by RULE-BASED detector (excluded from GNB):",
    ]
    for f in skipped_bbr:
        lines.append(f"  {os.path.basename(f)}")

    lines += ["", "CCAs in GNB model:"]
    for cls in sorted(le.classes_):
        lines.append(f"  {cls:<15}: {per_class.get(cls, 0)} segments")

    if removed:
        lines += ["", "Classes removed (too few segments):"]
        lines += [f"  {c}" for c in removed]

    if skipped_error:
        lines += ["", "Files skipped due to errors:"]
        lines += [f"  {f}: {e}" for f, e in skipped_error]

    if skipped_noseg:
        lines += ["", "Files with no usable segments:"]
        lines += [f"  {f}" for f in skipped_noseg]

    if cv_report:
        lines += ["", "Cross-validation classification report:", cv_report]

    path = os.path.join(out_dir, 'training_report.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(csv_dir=CSV_DIR, model_dir=MODEL_DIR,
          server_ip=SERVER_IP, rtt_s=RTT_S):

    os.makedirs(model_dir, exist_ok=True)

    # Step 1 ── build dataset ─────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1 — Building dataset")
    print(f"  CSV dir        : {csv_dir}")
    print(f"  Server IP      : {server_ip}")
    print(f"  RTT            : {rtt_s}s")
    print(f"  Rate-based excl: {RATE_BASED_CCAS}")
    print("=" * 60 + "\n")

    (X, y, skipped_bbr, skipped_error,
     skipped_noseg, per_class) = build_dataset(csv_dir, server_ip, rtt_s)

    if len(X) == 0:
        print("\nNo loss-based CCA data collected. "
              "Check CSV_DIR and that non-BBR CCAs are present.")
        sys.exit(1)

    # Step 2 ── validate ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — Validating dataset")
    print("=" * 60 + "\n")

    X, y, removed = validate_and_filter(X, y)

    missing = (ALL_CCAS - RATE_BASED_CCAS) - set(y)
    if missing:
        print(f"\n  CCAs with no data yet: {sorted(missing)}")
        print("  Run generate_dataset.sh more times to cover these.\n")

    print(f"\n  Segment counts per class:")
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        bar = '█' * min(cnt, 50)
        print(f"  {cls:<15}: {cnt:>4}  {bar}")

    # Step 3 ── feature cluster plots ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — Plotting feature clusters")
    print("=" * 60 + "\n")
    plot_feature_clusters_3d(X, y, model_dir)
    plot_feature_clusters_2d(X, y, model_dir)

    # Step 4 ── train GNB ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — Training Gaussian Naive Bayes (cross-validated)")
    print("=" * 60)
    gnb, le, cv_report = train_gnb(X, y)

    # Step 5 ── save ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 — Saving model and report")
    print("=" * 60 + "\n")

    joblib.dump(gnb, os.path.join(model_dir, 'gnb.pkl'))
    joblib.dump(le,  os.path.join(model_dir, 'label_encoder.pkl'))
    print(f"  Saved: {model_dir}/gnb.pkl")
    print(f"  Saved: {model_dir}/label_encoder.pkl")

    save_report(per_class, removed, skipped_error,
                skipped_noseg, skipped_bbr, le, cv_report, model_dir)

    print("\n  Done.")
    print(f"  GNB classes : {list(le.classes_)}")
    print(f"  BBR handled : rule-based detector (classify.py)\n")
    return gnb, le


if __name__ == '__main__':
    train()