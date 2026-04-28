"""
evaluate.py — Full evaluation of the Nebby classifier (dual-profile, all CCAs)
Paper reference: Nebby §4.1 (Table 3 — Confusion Matrix)

CHANGES FROM PREVIOUS VERSION:
  - Uses pair_traces() from train.py to pair 50ms + 100ms CSVs
  - Calls classify_trace_pair() for GNB (6D features) instead of
    single-profile 3D classification
  - BBR still evaluated via rule-based detector
  - Confusion matrix covers all CCAs (BBR via rule + rest via GNB)
  - Added per_class_accuracy.png bar chart

Usage:
    python3 evaluate.py

Outputs (saved to ../evaluation/):
    confusion_matrix.png
    confusion_matrix_counts.png
    bif_traces_eval.png
    confidence_histogram.png
    per_class_accuracy.png
    evaluation_report.txt
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

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import (extract_features,
                         extract_features_dual_profile)
from classify   import detect_bbr, _majority_vote
from train      import pair_traces

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR   = '../candidates-measurements'
MODEL_DIR = '../models'
OUT_DIR   = '../evaluation'
SERVER_IP = '10.0.0.1'

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
    gnb = joblib.load(os.path.join(MODEL_DIR, 'gnb.pkl'))
    le  = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    n_feat = gnb.theta_.shape[1]
    print(f"Loaded model")
    print(f"  GNB classes  : {list(le.classes_)}")
    print(f"  Feature dim  : {n_feat}D  "
          f"({'dual-profile' if n_feat==6 else 'single-profile'})")
    print(f"  BBR via rule : {RATE_BASED_CCAS}\n")
    return gnb, le


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREDICT ALL TRACES
# ══════════════════════════════════════════════════════════════════════════════

def _bbr_files(csv_dir):
    """Return list of BBR CSVs (unpaired — evaluated via rule detector)."""
    files = sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv')))
    bbr   = []
    for f in files:
        cc = re.search(r'cc-(\w+)_', os.path.basename(f))
        if cc and cc.group(1) in RATE_BASED_CCAS:
            bbr.append(f)
    return bbr


def predict_all(csv_dir, gnb, le):
    """
    Run full hybrid pipeline (rule + GNB) on all traces.
    Returns seg_results (segment level) and trace_results (trace level).
    """
    pairs, unpaired = pair_traces(csv_dir)
    bbr_files       = _bbr_files(csv_dir)
    n_feat          = gnb.theta_.shape[1]
    all_ccas        = set(le.classes_) | RATE_BASED_CCAS

    seg_results   = []
    trace_results = []

    print(f"  {'TRUE':<12} {'PRED':<12} {'CONF':>5}  {'SEGS':>4}  {'METHOD':<12}  FILE")
    print(f"  {'─'*12} {'─'*12} {'─'*5}  {'─'*4}  {'─'*12}  {'─'*40}")

    # ── BBR: rule-based ───────────────────────────────────────────────────────
    for fpath in bbr_files:
        fname      = os.path.basename(fpath)
        true_label = re.search(r'cc-(\w+)_', fname).group(1)

        try:
            t, bif       = compute_bif(fpath, SERVER_IP)
            t_s, bif_s   = smooth_bif(t, bif, 0.10)
            t_ss, bif_ss = remove_slow_start(t_s, bif_s)
            bbr_result   = detect_bbr(t_ss, bif_ss, 0.10)
        except Exception as e:
            print(f"  {true_label:<12} ERROR: {e}")
            continue

        pred_label = bbr_result if bbr_result else 'unknown'
        correct    = pred_label.startswith('bbr') and true_label.startswith('bbr')
        mark       = '✓' if correct else '✗'

        print(f"  {true_label:<12} {pred_label:<12} {'100%':>5}  {'─':>4}  "
              f"{'rule':<12}  {mark} {fname}")

        trace_results.append({
            'file': fname, 'true_label': true_label,
            'pred_label': pred_label, 'confidence': 1.0,
            'n_segments': 0, 'correct': correct,
            'method': 'bbr_rule', 't': t_ss, 'bif': bif_ss,
        })

    # ── Loss-based: dual-profile GNB ─────────────────────────────────────────
    for label, f50, f100 in pairs:
        fname50  = os.path.basename(f50)
        fname100 = os.path.basename(f100)

        try:
            if n_feat == 6:
                feats, n = extract_features_dual_profile(f50, f100, SERVER_IP)
            else:
                # Model is 3D — use single profile
                t, bif       = compute_bif(f50, SERVER_IP)
                t_s, bif_s   = smooth_bif(t, bif, 0.10)
                t_ss, bif_ss = remove_slow_start(t_s, bif_s)
                segs         = segment_bif(t_ss, bif_ss)
                feats        = extract_features(segs)
                n            = len(feats)
        except Exception as e:
            print(f"  {label:<12} ERROR: {e}  ({fname50})")
            continue

        # Load BiF for plotting (use 50ms trace)
        try:
            t_plot, bif_plot = compute_bif(f50, SERVER_IP)
            t_sp, bif_sp     = smooth_bif(t_plot, bif_plot, 0.10)
            t_ss_p, bif_ss_p = remove_slow_start(t_sp, bif_sp)
        except Exception:
            t_ss_p, bif_ss_p = np.array([0]), np.array([0])

        if n == 0:
            print(f"  {label:<12} {'unknown':<12} {'─':>5}     0  "
                  f"{'no_segments':<12}  ✗ {fname50}")
            trace_results.append({
                'file': fname50, 'true_label': label,
                'pred_label': 'unknown', 'confidence': 0.0,
                'n_segments': 0, 'correct': False,
                'method': 'no_segments',
                't': t_ss_p, 'bif': bif_ss_p,
            })
            continue

        seg_preds = gnb.predict(feats)
        seg_probs = gnb.predict_proba(feats)
        pred_label, confidence = _majority_vote(seg_preds, le)
        correct    = (pred_label == label)
        mark       = '✓' if correct else '✗'
        method     = f'GNB-{n_feat}D'

        print(f"  {label:<12} {pred_label:<12} {confidence:>4.0%}  {n:>4}  "
              f"{method:<12}  {mark} {fname50}")

        # Segment-level records
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
            'pred_label': pred_label, 'confidence': confidence,
            'n_segments': n, 'correct': correct,
            'method': method,
            't': t_ss_p, 'bif': bif_ss_p,
        })

    return seg_results, trace_results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TEXT REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_reports(seg_results, trace_results, le, out_dir):
    correct = sum(r['correct'] for r in trace_results)
    total   = len(trace_results)
    lines   = [
        "Nebby Evaluation Report — Dual Profile",
        "=" * 60,
        "",
        f"TRACE-LEVEL ACCURACY : {correct}/{total} = "
        f"{correct/total:.1%}" if total else "No traces",
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

    # Per-class
    lines += ["", "PER-CLASS TRACE ACCURACY:"]
    all_true = [r['true_label'] for r in trace_results]
    all_pred = [r['pred_label'] for r in trace_results]
    for cls in sorted(set(all_true)):
        idxs  = [i for i, t in enumerate(all_true) if t == cls]
        n_ok  = sum(all_pred[i] == cls for i in idxs)
        n_tot = len(idxs)
        bar   = '█' * n_ok + '░' * (n_tot - n_ok)
        lines.append(f"  {cls:<15}: {n_ok}/{n_tot} = "
                     f"{n_ok/n_tot:.0%}  {bar}"
                     if n_tot else f"  {cls:<15}: no data")

    # Segment-level
    known = [r for r in seg_results if r['true_enc'] >= 0]
    if known:
        y_true = [r['true_enc'] for r in known]
        y_pred = [r['pred_enc'] for r in known]
        seg_acc = accuracy_score(y_true, y_pred)
        lines += [
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

    path = os.path.join(out_dir, 'evaluation_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nSaved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(trace_results, out_dir):
    all_labels = sorted(set(
        [r['true_label'] for r in trace_results] +
        [r['pred_label'] for r in trace_results
         if r['pred_label'] != 'unknown']
    ))
    if not all_labels:
        return

    y_true = [r['true_label'] for r in trace_results]
    y_pred = [r['pred_label'] for r in trace_results]
    n      = len(all_labels)
    fs     = max(8, n * 0.8)

    for normalise, suffix, fmt in [
        ('true', 'confusion_matrix.png',        '.0%'),
        (None,   'confusion_matrix_counts.png', 'd'),
    ]:
        fig, ax = plt.subplots(figsize=(fs, fs * 0.85))
        cm = confusion_matrix(y_true, y_pred,
                              labels=all_labels,
                              normalize=normalise)
        im = ax.imshow(cm, cmap='Blues',
                       vmin=0, vmax=(1 if normalise else None))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True',      fontsize=10)
        ax.set_title(
            'Nebby — Confusion Matrix\n'
            '(trace-level, ' +
            ('row-normalised %' if normalise else 'raw counts') +
            ',  BBR via rule + others via 6D GNB)',
            fontsize=10,
        )

        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v > 0:
                    text = f'{v:.0%}' if normalise else str(int(v))
                    thresh = 0.5 if normalise else cm.max() * 0.5
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=7,
                            color='white' if v > thresh else 'black')

        plt.tight_layout()
        path = os.path.join(out_dir, suffix)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PER-CLASS ACCURACY BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_class_accuracy(trace_results, out_dir):
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

    overall = sum(r['correct'] for r in trace_results) / len(trace_results)
    ax.axhline(overall * 100, color='black', lw=1.5,
               linestyle='--', alpha=0.6, label=f'Overall {overall:.0%}')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel('Trace-level accuracy (%)', fontsize=10)
    ax.set_title('Per-CCA Accuracy  '
                 '(BBR via rule-based, others via 6D dual-profile GNB)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()

    path = os.path.join(out_dir, 'per_class_accuracy.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BiF TRACE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_bif_traces(trace_results, out_dir):
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
            f"conf={r['confidence']:.0%}  segs={r['n_segments']}  "
            f"[{r['method']}]",
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
# 7.  CONFIDENCE HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════

def plot_confidence_histogram(seg_results, le, out_dir):
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

            correct_conf   = [r['confidence'] for r in seg_results
                              if r['true_label'] == cls
                              and r['pred_label'] == cls]
            incorrect_conf = [r['confidence'] for r in seg_results
                              if r['true_label'] == cls
                              and r['pred_label'] != cls]

            if correct_conf:
                ax.hist(correct_conf, bins=bins, alpha=0.75,
                        color=color,    label=f'Correct ({len(correct_conf)})')
            if incorrect_conf:
                ax.hist(incorrect_conf, bins=bins, alpha=0.75,
                        color='tomato', label=f'Wrong ({len(incorrect_conf)})')

            ax.set_title(cls.upper(), fontsize=9)
            ax.set_xlabel("GNB confidence", fontsize=7)
            ax.set_ylabel("# Segments",     fontsize=7)
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
    print("  Nebby — Full Evaluation  (dual-profile, all CCAs)")
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