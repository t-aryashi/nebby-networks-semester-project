"""
evaluate_quic.py — Full evaluation of the Nebby QUIC classifier
Paper reference: Nebby §4.1 (Table 3 — Confusion Matrix), QUIC adaptation

HOW THIS RELATES TO evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
evaluate.py uses:
  compute_bif()                  from bif.py
  extract_features_dual_profile() from features.py
  pair_traces()                  from train.py

This file is structurally identical but uses:
  compute_bif_quic()             from quic_bif.py
  _get_quic_features_dual()      from train_quic.py
  pair_traces_quic()             from train_quic.py
  detect_bbr_quic()              from classify_quic.py  ← KEY FIX

WHY detect_bbr_quic() INSTEAD OF detect_bbr()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
detect_bbr() was tuned for TCP BiF (exact, from seq/ACK numbers).
QUIC BiF is estimated — same shape but different absolute scale.
The TCP thresholds caused all BBR QUIC traces to return None,
showing pred=unknown in the evaluation plot.
detect_bbr_quic() uses looser relative thresholds and periodicity
checks that match the staircase pattern visible in QUIC BiF plots.

Usage:
    python3 evaluate_quic.py

Outputs (saved to ../evaluation_quic/):
    confusion_matrix_quic.png
    confusion_matrix_counts_quic.png
    bif_traces_eval_quic.png
    confidence_histogram_quic.png
    per_class_accuracy_quic.png
    evaluation_report_quic.txt
"""

import os, sys, glob, re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))

# ── your existing modules (unchanged) ────────────────────────────────────────
from bif        import smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features
from classify   import _majority_vote

# ── QUIC-specific modules ─────────────────────────────────────────────────────
from quic_bif      import compute_bif_quic
from train_quic    import (pair_traces_quic,
                            _get_quic_features,
                            _get_quic_features_dual)
from classify_quic import detect_bbr_quic   # ← QUIC-tuned BBR detector

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR   = '../candidates-measurements-quic'
MODEL_DIR = '../models'
OUT_DIR   = '../evaluation_quic'
SERVER_IP = None   # None = auto-detect from CSV

RATE_BASED_CCAS = {'bbr', 'bbr2', 'bbr3'}

_PALETTE = [
    '#2196F3', '#e63946', '#FF9800', '#9C27B0', '#00BCD4',
    '#4CAF50', '#795548', '#607D8B', '#E91E63', '#FF5722',
    '#009688', '#FFC107', '#3F51B5', '#8BC34A', '#F44336',
    '#673AB7', '#555555',
]


def _color(cca, all_ccas):
    lst = sorted(all_ccas)
    idx = lst.index(cca) if cca in lst else -1
    return _PALETTE[idx % len(_PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """
    Load QUIC-specific model (gnb_quic.pkl) if it exists,
    otherwise fall back to the TCP model (gnb.pkl).
    """
    quic_gnb = os.path.join(MODEL_DIR, 'gnb_quic.pkl')
    quic_le  = os.path.join(MODEL_DIR, 'label_encoder_quic.pkl')
    tcp_gnb  = os.path.join(MODEL_DIR, 'gnb.pkl')
    tcp_le   = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    if os.path.exists(quic_gnb) and os.path.exists(quic_le):
        gnb = joblib.load(quic_gnb)
        le  = joblib.load(quic_le)
        model_note = "QUIC-specific (gnb_quic.pkl)"
    elif os.path.exists(tcp_gnb) and os.path.exists(tcp_le):
        gnb = joblib.load(tcp_gnb)
        le  = joblib.load(tcp_le)
        model_note = "TCP fallback (gnb.pkl) — run train_quic.py for better accuracy"
    else:
        raise FileNotFoundError(
            f"No model found in {MODEL_DIR}.\n"
            "Run  python3 train_quic.py  first."
        )

    n_feat = gnb.theta_.shape[1]
    print(f"Loaded model: {model_note}")
    print(f"  GNB classes  : {list(le.classes_)}")
    print(f"  Feature dim  : {n_feat}D  "
          f"({'dual-profile' if n_feat==6 else 'single-profile'})")
    print(f"  BBR via rule : detect_bbr_quic() (QUIC-tuned)\n")
    return gnb, le


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BBR FILES
# ══════════════════════════════════════════════════════════════════════════════

def _bbr_files_quic(csv_dir):
    """Return list of BBR QUIC CSVs (50ms profile only — rule detector)."""
    patterns = [
        os.path.join(csv_dir, '*_quic_*.csv'),
        os.path.join(csv_dir, '*.csv'),
    ]
    files = sorted(set(f for pat in patterns for f in glob.glob(pat)))
    bbr   = []
    for f in files:
        cc = re.search(r'cc-(\w+)[_.]', os.path.basename(f))
        if cc and cc.group(1) in RATE_BASED_CCAS:
            bbr.append(f)
    by_cca = defaultdict(list)
    for f in bbr:
        cc = re.search(r'cc-(\w+)[_.]', os.path.basename(f))
        if cc:
            by_cca[cc.group(1)].append(f)
    result = []
    for flist in by_cca.values():
        result.extend(flist[::2])   # even-indexed = 50ms runs
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREDICT ALL
# ══════════════════════════════════════════════════════════════════════════════

def predict_all_quic(csv_dir, gnb, le):
    """
    Run full hybrid pipeline (QUIC BBR rule + GNB) on all QUIC traces.
    Returns (seg_results, trace_results).
    """
    pairs, unpaired = pair_traces_quic(csv_dir)
    bbr_files       = _bbr_files_quic(csv_dir)
    n_feat          = gnb.theta_.shape[1]

    seg_results   = []
    trace_results = []

    print(f"  {'TRUE':<12} {'PRED':<12} {'CONF':>5}  {'SEGS':>4}  {'METHOD':<14}  FILE")
    print(f"  {'─'*12} {'─'*12} {'─'*5}  {'─'*4}  {'─'*14}  {'─'*40}")

    # ── BBR: QUIC rule-based ──────────────────────────────────────────────────
    for fpath in bbr_files:
        fname      = os.path.basename(fpath)
        m          = re.search(r'cc-(\w+)[_.]', fname)
        true_label = m.group(1) if m else 'bbr'

        try:
            t, bif       = compute_bif_quic(fpath, SERVER_IP)
            t_s, bif_s   = smooth_bif(t, bif, 0.10)
            t_ss, bif_ss = remove_slow_start(t_s, bif_s)
            # ← Use QUIC-tuned detector, not TCP detect_bbr()
            bbr_result   = detect_bbr_quic(t_ss, bif_ss, 0.10)
        except Exception as e:
            print(f"  {true_label:<12} ERROR: {e}")
            continue

        pred_label = bbr_result if bbr_result else 'unknown'
        correct    = pred_label.startswith('bbr') and true_label.startswith('bbr')
        mark       = '✓' if correct else '✗'

        print(f"  {true_label:<12} {pred_label:<12} {'100%':>5}  {'─':>4}  "
              f"{'quic_bbr_rule':<14}  {mark} {fname}")

        trace_results.append({
            'file': fname, 'true_label': true_label,
            'pred_label': pred_label, 'confidence': 1.0,
            'n_segments': 0, 'correct': correct,
            'method': 'quic_bbr_rule', 't': t_ss, 'bif': bif_ss,
        })

    # ── Loss-based: dual-profile GNB ─────────────────────────────────────────
    for label, f50, f100 in pairs:
        fname50  = os.path.basename(f50)
        fname100 = os.path.basename(f100)

        # BiF for plotting (50ms trace)
        try:
            t_plot, bif_plot = compute_bif_quic(f50, SERVER_IP)
            t_sp, bif_sp     = smooth_bif(t_plot, bif_plot, 0.10)
            t_ss_p, bif_ss_p = remove_slow_start(t_sp, bif_sp)
        except Exception:
            t_ss_p, bif_ss_p = np.array([0.0]), np.array([0.0])

        try:
            if n_feat == 6:
                feats, n = _get_quic_features_dual(f50, f100, SERVER_IP)
            else:
                feats = _get_quic_features(f50, rtt_s=0.10, server_ip=SERVER_IP)
                n     = len(feats)
        except Exception as e:
            print(f"  {label:<12} ERROR: {e}  ({fname50})")
            continue

        if n == 0:
            print(f"  {label:<12} {'unknown':<12} {'─':>5}     0  "
                  f"{'no_segments':<14}  ✗ {fname50}")
            trace_results.append({
                'file': fname50, 'true_label': label,
                'pred_label': 'unknown', 'confidence': 0.0,
                'n_segments': 0, 'correct': False,
                'method': 'no_segments', 't': t_ss_p, 'bif': bif_ss_p,
            })
            continue

        seg_preds        = gnb.predict(feats)
        seg_probs        = gnb.predict_proba(feats)
        pred_label, conf = _majority_vote(seg_preds, le)
        correct          = (pred_label == label)
        mark             = '✓' if correct else '✗'
        method           = f'GNB-{n_feat}D-QUIC'

        print(f"  {label:<12} {pred_label:<12} {conf:>4.0%}  {n:>4}  "
              f"{method:<14}  {mark} {fname50}")

        try:
            true_enc = le.transform([label])[0]
        except ValueError:
            true_enc = -1

        for i, (pe, probs) in enumerate(zip(seg_preds, seg_probs)):
            seg_results.append({
                'file':       fname50,
                'true_label': label,
                'pred_label': le.inverse_transform([pe])[0],
                'true_enc':   true_enc,
                'pred_enc':   pe,
                'confidence': probs.max(),
                'segment_id': i,
            })

        trace_results.append({
            'file': fname50, 'true_label': label,
            'pred_label': pred_label, 'confidence': conf,
            'n_segments': n, 'correct': correct,
            'method': method, 't': t_ss_p, 'bif': bif_ss_p,
        })

    return seg_results, trace_results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REPORTS + PLOTS  (identical to evaluate.py, filenames suffixed _quic)
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_reports_quic(seg_results, trace_results, le, out_dir):
    correct = sum(r['correct'] for r in trace_results)
    total   = len(trace_results)
    lines   = [
        "Nebby QUIC Evaluation Report — Dual Profile",
        "=" * 60,
        "",
        (f"TRACE-LEVEL ACCURACY : {correct}/{total} = "
         f"{correct/total:.1%}" if total else "No traces"),
        "",
        f"  {'FILE':<48} {'TRUE':<12} {'PRED':<12} {'CONF':>5}  OK?",
        f"  {'─'*48} {'─'*12} {'─'*12} {'─'*5}  {'─'*3}",
    ]
    for r in trace_results:
        mark = '✓' if r['correct'] else '✗'
        lines.append(
            f"  {r['file']:<48} {r['true_label']:<12} "
            f"{r['pred_label']:<12} {r['confidence']:>4.0%}  {mark}"
        )

    lines += ["", "PER-CLASS TRACE ACCURACY:"]
    all_true = [r['true_label'] for r in trace_results]
    all_pred = [r['pred_label'] for r in trace_results]
    for cls in sorted(set(all_true)):
        idxs  = [i for i, t in enumerate(all_true) if t == cls]
        n_ok  = sum(all_pred[i] == cls for i in idxs)
        n_tot = len(idxs)
        bar   = '█' * n_ok + '░' * (n_tot - n_ok)
        lines.append(
            f"  {cls:<15}: {n_ok}/{n_tot} = {n_ok/n_tot:.0%}  {bar}"
            if n_tot else f"  {cls:<15}: no data"
        )

    known = [r for r in seg_results if r['true_enc'] >= 0]
    if known:
        y_true  = [r['true_enc'] for r in known]
        y_pred  = [r['pred_enc'] for r in known]
        seg_acc = accuracy_score(y_true, y_pred)
        lines  += [
            "",
            f"SEGMENT-LEVEL ACCURACY (GNB only): {seg_acc:.1%}  "
            f"({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})",
            "",
            "SEGMENT CLASSIFICATION REPORT:",
            classification_report(y_true, y_pred,
                                  target_names=le.classes_,
                                  digits=3, zero_division=0),
        ]

    report = '\n'.join(lines)
    print("\n" + report)
    path = os.path.join(out_dir, 'evaluation_report_quic.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nSaved: {path}")


def plot_confusion_matrix_quic(trace_results, out_dir):
    all_labels = sorted(set(
        [r['true_label'] for r in trace_results] +
        [r['pred_label'] for r in trace_results if r['pred_label'] != 'unknown']
    ))
    if not all_labels:
        return

    y_true = [r['true_label'] for r in trace_results]
    y_pred = [r['pred_label'] for r in trace_results]
    n  = len(all_labels)
    fs = max(8, n * 0.8)

    for normalise, suffix, _ in [
        ('true', 'confusion_matrix_quic.png',        '.0%'),
        (None,   'confusion_matrix_counts_quic.png', 'd'),
    ]:
        fig, ax = plt.subplots(figsize=(fs, fs * 0.85))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize=normalise)
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=(1 if normalise else None))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True',      fontsize=10)
        ax.set_title(
            'Nebby QUIC — Confusion Matrix\n'
            '(trace-level, ' +
            ('row-normalised %' if normalise else 'raw counts') +
            ',  BBR via QUIC rule + others via 6D GNB)',
            fontsize=10,
        )
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v > 0:
                    text   = f'{v:.0%}' if normalise else str(int(v))
                    thresh = 0.5 if normalise else cm.max() * 0.5
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=7,
                            color='white' if v > thresh else 'black')
        plt.tight_layout()
        path = os.path.join(out_dir, suffix)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


def plot_per_class_accuracy_quic(trace_results, out_dir):
    all_true = [r['true_label'] for r in trace_results]
    all_pred = [r['pred_label'] for r in trace_results]
    classes  = sorted(set(all_true))
    all_cls  = set(all_true)

    accs, totals = [], []
    for cls in classes:
        idxs  = [i for i, t in enumerate(all_true) if t == cls]
        n_ok  = sum(all_pred[i] == cls for i in idxs)
        n_tot = len(idxs)
        accs.append(n_ok / n_tot if n_tot else 0)
        totals.append(n_tot)

    colors = [_color(c, all_cls) for c in classes]
    x      = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(max(10, len(classes)), 5))
    bars = ax.bar(x, [a * 100 for a in accs],
                  color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, acc, tot in zip(bars, accs, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{acc:.0%}\n(n={tot})',
                ha='center', va='bottom', fontsize=8)

    overall = sum(r['correct'] for r in trace_results) / max(len(trace_results), 1)
    ax.axhline(overall * 100, color='black', lw=1.5,
               linestyle='--', alpha=0.6, label=f'Overall {overall:.0%}')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel('Trace-level accuracy (%)', fontsize=10)
    ax.set_title('QUIC Per-CCA Accuracy  '
                 '(BBR via QUIC rule-based, others via 6D GNB)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    path = os.path.join(out_dir, 'per_class_accuracy_quic.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_bif_traces_quic(trace_results, out_dir):
    n = len(trace_results)
    if n == 0:
        return
    all_cls = set(r['true_label'] for r in trace_results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(6 * cols, 4 * rows))
    gs   = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.6, wspace=0.35)

    for idx, r in enumerate(trace_results):
        ax    = fig.add_subplot(gs[idx // cols, idx % cols])
        t     = r['t']
        bif   = r['bif']
        pred  = r['pred_label']
        true  = r['true_label']
        color = _color(pred, all_cls)
        mark  = '✓' if r['correct'] else '✗'

        bif_roll = (pd.Series(bif)
                    .rolling(15, center=True, min_periods=1)
                    .mean().values)

        ax.fill_between(t, 0, bif / 1024, alpha=0.10, color=color)
        ax.plot(t, bif      / 1024, color=color, lw=0.5, alpha=0.4)
        ax.plot(t, bif_roll / 1024, color=color, lw=1.8,
                label=f"pred={pred} {mark}")
        ax.set_title(
            f"true={true}  pred={pred}  {mark}\n"
            f"conf={r['confidence']:.0%}  segs={r['n_segments']}  "
            f"[{r['method']}]  QUIC",
            fontsize=7,
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    fig.suptitle("Nebby QUIC — BiF Traces (coloured by predicted CCA)",
                 fontsize=12, y=1.01)
    path = os.path.join(out_dir, 'bif_traces_eval_quic.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_confidence_histogram_quic(seg_results, le, out_dir):
    classes = list(le.classes_)
    n_cls   = len(classes)
    if n_cls == 0:
        return

    cols = min(n_cls, 4)
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5 * cols, 3.5 * rows),
                              squeeze=False)
    bins = np.linspace(0, 1, 21)

    for r_idx in range(rows):
        for c_idx in range(cols):
            idx = r_idx * cols + c_idx
            ax  = axes[r_idx][c_idx]
            if idx >= n_cls:
                ax.set_visible(False)
                continue
            cls   = classes[idx]
            color = _color(cls, set(classes))
            cc    = [r['confidence'] for r in seg_results
                     if r['true_label']==cls and r['pred_label']==cls]
            ic    = [r['confidence'] for r in seg_results
                     if r['true_label']==cls and r['pred_label']!=cls]
            if cc:
                ax.hist(cc, bins=bins, alpha=0.75, color=color,
                        label=f'Correct ({len(cc)})')
            if ic:
                ax.hist(ic, bins=bins, alpha=0.75, color='tomato',
                        label=f'Wrong ({len(ic)})')
            ax.set_title(cls.upper() + ' (QUIC)', fontsize=9)
            ax.set_xlabel("GNB confidence", fontsize=7)
            ax.set_ylabel("# Segments",     fontsize=7)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Nebby QUIC — GNB Confidence by Class", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'confidence_histogram_quic.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Nebby — QUIC Evaluation  (dual-profile, all CCAs)")
    print("=" * 60 + "\n")

    gnb, le = load_model()

    print(f"Scanning {CSV_DIR} ...\n")
    seg_results, trace_results = predict_all_quic(CSV_DIR, gnb, le)

    if not trace_results:
        print("\nNo results. Check CSV_DIR contains QUIC CSVs.")
        sys.exit(1)

    print("\nGenerating reports and plots ...")
    print_and_save_reports_quic(seg_results, trace_results, le, OUT_DIR)
    plot_confusion_matrix_quic(trace_results, OUT_DIR)
    plot_per_class_accuracy_quic(trace_results, OUT_DIR)
    plot_bif_traces_quic(trace_results, OUT_DIR)
    if seg_results:
        plot_confidence_histogram_quic(seg_results, le, OUT_DIR)

    print(f"\nAll QUIC outputs saved to  {OUT_DIR}/")