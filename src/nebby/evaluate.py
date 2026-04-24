"""
evaluate.py — Full evaluation of the Nebby classifier for all 17 CCAs
Paper reference: Nebby §4.1 (Table 3 — Confusion Matrix)

Evaluation uses the same hybrid pipeline as classify.py:
  BBR traces   → rule-based detector (detect_bbr)
  All others   → GNB on polynomial features

Usage:
    python3 evaluate.py

Outputs (saved to ../evaluation/):
    confusion_matrix.png          row-normalised % matrix (mirrors Table 3)
    confusion_matrix_counts.png   raw counts version
    bif_traces_eval.png           BiF subplot per CSV, coloured by prediction
    confidence_histogram.png      GNB confidence: correct vs wrong per class
    per_class_accuracy.png        bar chart of accuracy per CCA
    evaluation_report.txt         full text report
"""

import os, sys, glob, re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features
from classify   import detect_bbr

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR   = '../candidates-measurements'
MODEL_DIR = '../models'
OUT_DIR   = '../evaluation'
SERVER_IP = '10.0.0.1'
RTT_S     = 0.1

# Rate-based CCAs evaluated via rule detector, not GNB
RATE_BASED_CCAS = {'bbr', 'bbr2', 'bbr3'}

# Colour palette — one colour per CCA (up to 17)
_PALETTE = [
    '#e63946', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4',
    '#4CAF50', '#795548', '#607D8B', '#E91E63', '#FF5722',
    '#009688', '#FFC107', '#3F51B5', '#8BC34A', '#F44336',
    '#673AB7', '#555555',
]


def _color(cca, classes):
    idx = sorted(classes).index(cca) if cca in classes else -1
    return _PALETTE[idx % len(_PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    gnb_path = os.path.join(MODEL_DIR, 'gnb.pkl')
    le_path  = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if not os.path.exists(gnb_path) or not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}.\n"
            "Run  python3 train.py  first."
        )
    gnb = joblib.load(gnb_path)
    le  = joblib.load(le_path)
    print(f"Loaded model")
    print(f"  GNB classes  : {list(le.classes_)}")
    print(f"  BBR via rule : {RATE_BASED_CCAS}\n")
    return gnb, le


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREDICT ALL TRACES  (hybrid: rule-based for BBR, GNB for others)
# ══════════════════════════════════════════════════════════════════════════════

def predict_all(csv_dir, gnb, le):
    """
    Run the full hybrid pipeline on every CSV.

    Returns
    -------
    seg_results   : list of dicts — one per segment (loss-based CCAs only)
    trace_results : list of dicts — one per CSV file (all CCAs)
    """
    files = sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv')))
    if not files:
        raise FileNotFoundError(f"No *_tcp.csv files found in {csv_dir}")

    seg_results   = []
    trace_results = []

    all_classes = set(le.classes_) | RATE_BASED_CCAS

    print(f"  {'FILE':<52} {'TRUE':<12} {'PRED':<12} {'CONF':>5}  SEGS  OK?")
    print(f"  {'─'*52} {'─'*12} {'─'*12} {'─'*5}  {'─'*4}  {'─'*3}")

    for fpath in files:
        fname    = os.path.basename(fpath)
        cc_match = re.search(r'cc-(\w+)_', fname)
        if not cc_match:
            continue
        true_label = cc_match.group(1)

        try:
            t, bif         = compute_bif(fpath, SERVER_IP)
            t_s, bif_s     = smooth_bif(t, bif, RTT_S)
            t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
        except Exception as e:
            print(f"  {fname:<52} {true_label:<12} {'ERROR':<12} {'─':>5}  {'─':>4}  ✗  ({e})")
            continue

        # ── BBR: rule-based ───────────────────────────────────────────────
        if true_label in RATE_BASED_CCAS:
            bbr_result = detect_bbr(t_ss, bif_ss, RTT_S)
            pred_label = bbr_result if bbr_result else 'unknown'
            confidence = 1.0 if bbr_result else 0.0
            n_segs     = 0
            method     = 'bbr_rule'
            correct    = (pred_label == true_label or
                          (bbr_result == 'bbr' and true_label.startswith('bbr')))

            mark = '✓' if correct else '✗'
            print(f"  {fname:<52} {true_label:<12} {pred_label:<12} "
                  f"{confidence:>4.0%}  {'─':>4}  {mark}  [rule]")

            trace_results.append({
                'file': fname, 'true_label': true_label,
                'pred_label': pred_label, 'confidence': confidence,
                'n_segments': n_segs, 'correct': correct,
                'method': method, 't': t_ss, 'bif': bif_ss,
            })
            continue

        # ── Loss-based: GNB ───────────────────────────────────────────────
        segments = segment_bif(t_ss, bif_ss)
        feats    = extract_features(segments)

        if len(feats) == 0:
            print(f"  {fname:<52} {true_label:<12} {'unknown':<12} "
                  f"{'─':>5}     0  ✗  [no segs]")
            trace_results.append({
                'file': fname, 'true_label': true_label,
                'pred_label': 'unknown', 'confidence': 0.0,
                'n_segments': 0, 'correct': False,
                'method': 'no_segments', 't': t_ss, 'bif': bif_ss,
            })
            continue

        seg_preds = gnb.predict(feats)
        seg_probs = gnb.predict_proba(feats)

        # Majority vote across segments
        unique, counts = np.unique(seg_preds, return_counts=True)
        best_enc       = unique[np.argmax(counts)]
        confidence     = counts.max() / counts.sum()
        pred_label     = le.inverse_transform([best_enc])[0]
        correct        = (pred_label == true_label)

        # Segment-level records
        try:
            true_enc = le.transform([true_label])[0]
        except ValueError:
            true_enc = -1   # true label not in GNB (e.g. newly added CCA)

        for i, (pe, probs) in enumerate(zip(seg_preds, seg_probs)):
            seg_results.append({
                'file':       fname,
                'true_label': true_label,
                'pred_label': le.inverse_transform([pe])[0],
                'true_enc':   true_enc,
                'pred_enc':   pe,
                'confidence': probs.max(),
                'segment_id': i,
            })

        mark = '✓' if correct else '✗'
        print(f"  {fname:<52} {true_label:<12} {pred_label:<12} "
              f"{confidence:>4.0%}  {len(feats):>4}  {mark}  [GNB]")

        trace_results.append({
            'file': fname, 'true_label': true_label,
            'pred_label': pred_label, 'confidence': confidence,
            'n_segments': len(feats), 'correct': correct,
            'method': 'gnb', 't': t_ss, 'bif': bif_ss,
        })

    return seg_results, trace_results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TEXT REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_reports(seg_results, trace_results, le, out_dir):
    lines = ["Nebby Evaluation Report", "=" * 60, ""]

    # ── trace-level ───────────────────────────────────────────────────────
    correct = sum(r['correct'] for r in trace_results)
    total   = len(trace_results)
    acc_str = f"{correct/total:.1%}" if total else "N/A"

    lines += [
        f"TRACE-LEVEL ACCURACY : {correct}/{total} = {acc_str}",
        "",
        f"  {'FILE':<50} {'TRUE':<12} {'PRED':<12} {'CONF':>5}  OK?",
        f"  {'─'*50} {'─'*12} {'─'*12} {'─'*5}  {'─'*3}",
    ]
    for r in trace_results:
        mark = '✓' if r['correct'] else '✗'
        lines.append(
            f"  {r['file']:<50} {r['true_label']:<12} "
            f"{r['pred_label']:<12} {r['confidence']:>4.0%}  {mark}"
        )

    # ── per-class trace accuracy ──────────────────────────────────────────
    lines += ["", "PER-CLASS TRACE ACCURACY:"]
    all_true = [r['true_label'] for r in trace_results]
    all_pred = [r['pred_label'] for r in trace_results]
    for cls in sorted(set(all_true)):
        idxs  = [i for i, t in enumerate(all_true) if t == cls]
        n_ok  = sum(all_pred[i] == cls for i in idxs)
        n_tot = len(idxs)
        lines.append(f"  {cls:<15}: {n_ok}/{n_tot} = "
                     f"{n_ok/n_tot:.0%}" if n_tot else f"  {cls:<15}: no data")

    # ── segment-level (GNB only) ──────────────────────────────────────────
    known_segs = [r for r in seg_results if r['true_enc'] >= 0]
    if known_segs:
        y_true = [r['true_enc'] for r in known_segs]
        y_pred = [r['pred_enc'] for r in known_segs]
        seg_acc = accuracy_score(y_true, y_pred)
        lines += [
            "",
            f"SEGMENT-LEVEL ACCURACY (GNB only): {seg_acc:.1%}  "
            f"({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})",
            "",
            "SEGMENT-LEVEL CLASSIFICATION REPORT:",
            classification_report(
                y_true, y_pred,
                target_names=le.classes_,
                digits=3, zero_division=0,
            ),
        ]

    report_text = '\n'.join(lines)
    print("\n" + report_text)

    path = os.path.join(out_dir, 'evaluation_report.txt')
    with open(path, 'w') as f:
        f.write(report_text)
    print(f"\nSaved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONFUSION MATRIX  (trace-level, all CCAs including BBR)
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(trace_results, out_dir):
    """
    Trace-level confusion matrix covering ALL CCAs (BBR via rule + rest via GNB).
    This is the closest equivalent to Table 3 in the paper.
    """
    all_labels = sorted(set(
        [r['true_label'] for r in trace_results] +
        [r['pred_label'] for r in trace_results if r['pred_label'] != 'unknown']
    ))
    if not all_labels:
        return

    y_true = [r['true_label'] for r in trace_results]
    y_pred = [r['pred_label'] for r in trace_results]

    n = len(all_labels)
    # Scale figure with number of classes
    fig_size = max(8, n * 0.8)

    # ── normalised (%) ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize='true')
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True',      fontsize=10)
    ax.set_title('Nebby — Confusion Matrix\n'
                 '(trace-level, row-normalised %,  '
                 'BBR via rule + others via GNB)',
                 fontsize=11)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, f'{val:.0%}',
                        ha='center', va='center', fontsize=7,
                        color='white' if val > 0.5 else 'black')

    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ── raw counts ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    cm_raw  = confusion_matrix(y_true, y_pred, labels=all_labels)
    im2     = ax.imshow(cm_raw, cmap='Blues')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True',      fontsize=10)
    ax.set_title('Nebby — Confusion Matrix (raw counts)', fontsize=11)

    for i in range(n):
        for j in range(n):
            if cm_raw[i, j] > 0:
                ax.text(j, i, str(cm_raw[i, j]),
                        ha='center', va='center', fontsize=8,
                        color='white' if cm_raw[i, j] > cm_raw.max()*0.5
                        else 'black')

    plt.tight_layout()
    path2 = os.path.join(out_dir, 'confusion_matrix_counts.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PER-CLASS ACCURACY BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_class_accuracy(trace_results, out_dir):
    """Bar chart showing trace-level accuracy per CCA."""
    all_true = [r['true_label'] for r in trace_results]
    all_pred = [r['pred_label'] for r in trace_results]
    classes  = sorted(set(all_true))
    all_cls  = set(all_true)

    accs   = []
    totals = []
    for cls in classes:
        idxs  = [i for i, t in enumerate(all_true) if t == cls]
        n_ok  = sum(all_pred[i] == cls for i in idxs)
        n_tot = len(idxs)
        accs.append(n_ok / n_tot if n_tot else 0)
        totals.append(n_tot)

    colors = [_color(c, all_cls) for c in classes]
    x      = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.8), 5))
    bars = ax.bar(x, [a * 100 for a in accs],
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Annotate with count and accuracy
    for bar, acc, tot in zip(bars, accs, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{acc:.0%}\n(n={tot})',
                ha='center', va='bottom', fontsize=8)

    ax.axhline(100 * sum(r['correct'] for r in trace_results) / len(trace_results),
               color='black', lw=1.5, linestyle='--', alpha=0.6,
               label='Overall accuracy')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel('Trace-level accuracy (%)', fontsize=10)
    ax.set_title('Per-CCA Classification Accuracy\n'
                 '(BBR via rule-based detector, others via GNB)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()

    path = os.path.join(out_dir, 'per_class_accuracy.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BiF TRACE PLOTS  (one subplot per CSV)
# ══════════════════════════════════════════════════════════════════════════════

def plot_bif_traces(trace_results, out_dir):
    """One subplot per CSV, coloured by predicted CCA, ✓/✗ marked."""
    n = len(trace_results)
    if n == 0:
        return

    all_cls = set(r['true_label'] for r in trace_results)
    cols    = min(n, 4)
    rows    = (n + cols - 1) // cols
    fig     = plt.figure(figsize=(6 * cols, 4 * rows))
    gs      = gridspec.GridSpec(rows, cols, figure=fig,
                                hspace=0.6, wspace=0.35)

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
            f"conf={r['confidence']:.0%}  segs={r['n_segments']}  [{r['method']}]",
            fontsize=7,
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    fig.suptitle("Nebby — BiF Traces (coloured by predicted CCA)",
                 fontsize=12, y=1.01)

    path = os.path.join(out_dir, 'bif_traces_eval.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CONFIDENCE HISTOGRAM  (GNB segments only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_confidence_histogram(seg_results, le, out_dir):
    """GNB prediction confidence: correct vs wrong segments per class."""
    classes = list(le.classes_)
    n_cls   = len(classes)
    if n_cls == 0:
        return

    cols = min(n_cls, 4)
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5 * cols, 3.5 * rows),
                              sharey=False, squeeze=False)
    bins = np.linspace(0, 1, 21)

    for r_idx in range(rows):
        for c_idx in range(cols):
            idx = r_idx * cols + c_idx
            ax  = axes[r_idx][c_idx]
            if idx >= n_cls:
                ax.set_visible(False)
                continue
            cls = classes[idx]
            color = _color(cls, set(classes))

            correct_conf   = [r['confidence'] for r in seg_results
                              if r['true_label'] == cls
                              and r['pred_label'] == cls]
            incorrect_conf = [r['confidence'] for r in seg_results
                              if r['true_label'] == cls
                              and r['pred_label'] != cls]

            if correct_conf:
                ax.hist(correct_conf, bins=bins, alpha=0.75,
                        color=color,   label=f'Correct ({len(correct_conf)})')
            if incorrect_conf:
                ax.hist(incorrect_conf, bins=bins, alpha=0.75,
                        color='tomato', label=f'Wrong ({len(incorrect_conf)})')

            ax.set_title(cls.upper(), fontsize=9)
            ax.set_xlabel("GNB confidence", fontsize=7)
            ax.set_ylabel("# Segments", fontsize=7)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Nebby — GNB Confidence by Class", fontsize=12)
    plt.tight_layout()

    path = os.path.join(out_dir, 'confidence_histogram.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Nebby — Full Evaluation  (all 17 CCAs)")
    print("=" * 60 + "\n")

    gnb, le = load_model()

    print(f"Scanning {CSV_DIR} ...\n")
    seg_results, trace_results = predict_all(CSV_DIR, gnb, le)

    if not trace_results:
        print("\nNo results. Check CSV_DIR and SERVER_IP.")
        sys.exit(1)

    print("\nGenerating reports and plots ...")
    print_and_save_reports(seg_results, trace_results, le, OUT_DIR)
    plot_confusion_matrix(trace_results, OUT_DIR)
    plot_per_class_accuracy(trace_results, OUT_DIR)
    plot_bif_traces(trace_results, OUT_DIR)
    if seg_results:
        plot_confidence_histogram(seg_results, le, OUT_DIR)

    print(f"\nAll outputs saved to  {OUT_DIR}/")