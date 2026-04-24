"""
stream_classify.py — Online / Streaming CCA Classification
Novel contribution beyond Nebby (SIGCOMM '24)

The original paper processes a full completed trace offline.
This module simulates real-time classification using a sliding window
over the BiF trace — emulating what a network operator would see
mid-connection, before the download finishes.

Research questions answered:
  1. How many seconds of trace are needed before prediction stabilises?
  2. How quickly does the classifier converge to the correct CCA?
  3. Does confidence grow monotonically or oscillate?
  4. What is the earliest reliable detection time per CCA?

Usage:
    python3 stream_classify.py  <path_to_tcp.csv>  [--window 15] [--step 3]

Outputs (saved to ../evaluation/streaming/):
    stream_<filename>_timeline.png   — prediction over time
    stream_<filename>_confidence.png — confidence over time
    stream_summary.csv               — convergence stats for all traces
"""

import os, sys, glob, re, argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features
from classify   import detect_bbr

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = '../models'
CSV_DIR    = '../candidates-measurements'
OUT_DIR    = '../evaluation/streaming'
SERVER_IP  = '10.0.0.1'
RTT_S      = 0.1    # seconds

# Sliding window defaults (paper §3.3 uses ~18s minimum traces)
DEFAULT_WINDOW_S = 15.0   # seconds of BiF per window
DEFAULT_STEP_S   =  3.0   # slide forward by this much each step

COLORS = {'cubic': 'steelblue', 'reno': 'tomato',
          'bbr': 'seagreen',    'unknown': 'grey'}


# ══════════════════════════════════════════════════════════════════════════════
# CORE: sliding window predictor
# ══════════════════════════════════════════════════════════════════════════════

def sliding_window_classify(t, bif, gnb, le,
                             window_s=DEFAULT_WINDOW_S,
                             step_s=DEFAULT_STEP_S,
                             rtt_s=RTT_S):
    """
    Classify the CCA at each time step using only BiF data seen so far.

    At each position t_now, we use a window [t_now - window_s, t_now]
    of BiF data — mimicking a real-time observer who only has the
    last `window_s` seconds of the connection.

    Parameters
    ----------
    t, bif    : full smoothed BiF trace (post slow-start removal)
    gnb, le   : trained model + label encoder
    window_s  : width of the sliding window in seconds
    step_s    : how far to advance the window each step

    Returns
    -------
    results : list of dicts, one per time step, containing:
        t_now       : current time position (s)
        label       : predicted CCA string
        confidence  : fraction of segments agreeing  [0, 1]
        n_segments  : number of usable segments in this window
        method      : 'bbr_rule' | 'gnb' | 'unknown'
    """
    results  = []
    t_start  = t[0]
    t_end    = t[-1]

    # First window starts when we have enough data
    pos = t_start + window_s

    while pos <= t_end + step_s:
        pos = min(pos, t_end)

        # Extract window
        mask    = (t >= pos - window_s) & (t <= pos)
        t_win   = t[mask]
        bif_win = bif[mask]

        result = _classify_window(t_win, bif_win, gnb, le, rtt_s)
        result['t_now'] = pos
        results.append(result)

        if pos >= t_end:
            break
        pos += step_s

    return results


def _classify_window(t_win, bif_win, gnb, le, rtt_s):
    """Classify a single window of BiF data."""

    # Need at least 5 seconds to do anything meaningful
    if len(t_win) < 10 or (t_win[-1] - t_win[0]) < 2.0:
        return {'label': 'unknown', 'confidence': 0.0,
                'n_segments': 0, 'method': 'unknown'}

    # 1. BBR rule-based check on this window
    bbr = detect_bbr(t_win, bif_win, rtt_s)
    if bbr is not None:
        return {'label': 'bbr', 'confidence': 1.0,
                'n_segments': 0, 'method': 'bbr_rule'}

    # 2. Segment and classify with GNB
    # Note: we do NOT call remove_slow_start here because the window
    # is already in the congestion-avoidance phase (slow start removed
    # from the full trace before calling this function)
    segments = segment_bif(t_win, bif_win)
    feats    = extract_features(segments)

    if len(feats) == 0:
        return {'label': 'unknown', 'confidence': 0.0,
                'n_segments': 0, 'method': 'unknown'}

    preds          = gnb.predict(feats)
    probs          = gnb.predict_proba(feats)
    unique, counts = np.unique(preds, return_counts=True)
    best_enc       = unique[np.argmax(counts)]
    confidence     = counts.max() / counts.sum()
    label          = le.inverse_transform([best_enc])[0]

    # Mean confidence across segments (more informative than just vote share)
    mean_prob = np.mean(probs.max(axis=1))

    return {
        'label':      label,
        'confidence': confidence,
        'mean_prob':  mean_prob,
        'n_segments': len(feats),
        'method':     'gnb',
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_convergence(results, true_label, stable_window=3):
    """
    Find when the streaming classifier first converges to the correct answer
    and stays there for `stable_window` consecutive steps.

    Parameters
    ----------
    results      : list of dicts from sliding_window_classify
    true_label   : ground truth CCA string
    stable_window: number of consecutive correct steps = "converged"

    Returns
    -------
    dict with convergence stats
    """
    labels = [r['label'] for r in results]
    times  = [r['t_now'] for r in results]
    confs  = [r['confidence'] for r in results]

    # Overall accuracy across all windows
    correct     = [l == true_label for l in labels]
    accuracy    = sum(correct) / len(correct) if correct else 0.0

    # First stable convergence
    conv_time   = None
    conv_idx    = None
    for i in range(len(labels) - stable_window + 1):
        window_labels = labels[i:i + stable_window]
        if all(l == true_label for l in window_labels):
            conv_time = times[i]
            conv_idx  = i
            break

    # Confidence at convergence
    conv_conf = confs[conv_idx] if conv_idx is not None else None

    # Most common prediction (should be true_label if classifier works)
    most_common = Counter(labels).most_common(1)[0][0]

    return {
        'true_label':      true_label,
        'most_common_pred': most_common,
        'overall_accuracy': accuracy,
        'convergence_time': conv_time,
        'convergence_conf': conv_conf,
        'total_windows':   len(results),
        'correct_windows': sum(correct),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_timeline(results, true_label, fname, bif_t, bif_vals, out_dir,
                  window_s):
    """
    Two-panel plot:
      Top:    BiF trace (background) + shaded prediction colour per window
      Bottom: Confidence over time
    """
    times  = np.array([r['t_now']      for r in results])
    labels = [r['label']               for r in results]
    confs  = np.array([r['confidence'] for r in results])
    n_segs = np.array([r['n_segments'] for r in results])

    fig, (ax_bif, ax_conf) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True,
    )

    # ── top panel: BiF + coloured prediction bands ────────────────────────
    ax_bif.plot(bif_t, bif_vals / 1024,
                color='black', lw=0.8, alpha=0.4, label='BiF (smoothed)')

    for i, (t_now, label) in enumerate(zip(times, labels)):
        t_left  = t_now - window_s
        color   = COLORS.get(label, 'grey')
        correct = (label == true_label)
        alpha   = 0.25 if correct else 0.10
        ax_bif.axvspan(t_left, t_now, alpha=alpha, color=color,
                       linewidth=0)

    # Legend patches
    patches = [mpatches.Patch(color=COLORS.get(c, 'grey'), alpha=0.5, label=c)
               for c in ['cubic', 'reno', 'bbr', 'unknown']]
    patches.append(mpatches.Patch(color='white', label=f'true={true_label}',
                                  edgecolor='black'))
    ax_bif.legend(handles=patches, fontsize=8, loc='upper right')
    ax_bif.set_ylabel("KB in flight", fontsize=9)
    ax_bif.set_title(
        f"Streaming Classification — {fname}\n"
        f"True CCA: {true_label.upper()}   "
        f"Window: {window_s}s",
        fontsize=10,
    )
    ax_bif.grid(True, alpha=0.25)

    # Mark convergence
    for i in range(len(labels) - 2):
        if (labels[i] == true_label and
                labels[i+1] == true_label and
                labels[i+2] == true_label):
            ax_bif.axvline(times[i], color='black', lw=1.5,
                           linestyle='--', alpha=0.7, label='converged')
            ax_bif.text(times[i] + 0.5,
                        ax_bif.get_ylim()[1] * 0.9,
                        f'converged\nt={times[i]:.0f}s',
                        fontsize=7, color='black')
            break

    # ── bottom panel: confidence + n_segments ─────────────────────────────
    point_colors = [COLORS.get(l, 'grey') for l in labels]

    ax_conf.plot(times, confs, color='black', lw=1.0, alpha=0.4)
    ax_conf.scatter(times, confs, c=point_colors, s=30, zorder=3,
                    label='confidence (colour = prediction)')

    # Shade correct/wrong
    for i, (t_now, label, conf) in enumerate(zip(times, labels, confs)):
        color = 'green' if label == true_label else 'red'
        ax_conf.axvline(t_now, color=color, alpha=0.08, lw=6)

    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("Confidence", fontsize=9)
    ax_conf.set_xlabel("Time (s)",   fontsize=9)
    ax_conf.axhline(0.5, color='grey', lw=0.8, linestyle=':', alpha=0.7)
    ax_conf.grid(True, alpha=0.25)
    ax_conf.legend(fontsize=8)

    plt.tight_layout()
    safe_name = fname.replace('.csv', '').replace(' ', '_')
    path = os.path.join(out_dir, f"stream_{safe_name}_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_all_convergence(all_stats, out_dir):
    """
    Summary plot: convergence time per trace, grouped by CCA.
    Answers: 'Which CCA converges fastest?'
    """
    df = pd.DataFrame(all_stats)
    df = df.dropna(subset=['convergence_time'])

    if df.empty:
        print("  No convergence data to plot (no trace converged).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── convergence time per CCA ──────────────────────────────────────────
    ax = axes[0]
    for cca in sorted(df['true_label'].unique()):
        sub = df[df['true_label'] == cca]['convergence_time']
        ax.bar(cca, sub.mean(),
               color=COLORS.get(cca, 'grey'), alpha=0.8,
               yerr=sub.std() if len(sub) > 1 else 0,
               capsize=5, label=cca)
        for val in sub.values:
            ax.scatter(cca, val,
                       color=COLORS.get(cca, 'grey'),
                       edgecolors='black', s=50, zorder=3)

    ax.set_ylabel("Convergence time (s)", fontsize=10)
    ax.set_title("Time to stable correct prediction\n(3 consecutive correct windows)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ── overall accuracy per trace ─────────────────────────────────────────
    ax2 = axes[1]
    for cca in sorted(df['true_label'].unique()):
        sub = df[df['true_label'] == cca]['overall_accuracy']
        ax2.bar(cca, sub.mean() * 100,
                color=COLORS.get(cca, 'grey'), alpha=0.8,
                yerr=sub.std() * 100 if len(sub) > 1 else 0,
                capsize=5)
        for val in sub.values:
            ax2.scatter(cca, val * 100,
                        color=COLORS.get(cca, 'grey'),
                        edgecolors='black', s=50, zorder=3)

    ax2.set_ylim(0, 110)
    ax2.set_ylabel("% windows correctly classified", fontsize=10)
    ax2.set_title("Window-level accuracy across full trace", fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Streaming Nebby — Convergence Analysis", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'stream_convergence_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_window_size_sensitivity(csv_path, gnb, le, out_dir,
                                 server_ip=SERVER_IP, rtt_s=RTT_S):
    """
    Novel experiment: how does window size affect convergence time?
    Sweeps window_s = [5, 10, 15, 20, 30] on a single trace.
    """
    fname      = os.path.basename(csv_path)
    cc_match   = re.search(r'cc-(\w+)_', fname)
    true_label = cc_match.group(1) if cc_match else 'unknown'

    t, bif       = compute_bif(csv_path, server_ip)
    t_s, bif_s   = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss = remove_slow_start(t_s, bif_s)

    windows      = [5, 8, 10, 15, 20, 30]
    conv_times   = []
    accuracies   = []

    for w in windows:
        results = sliding_window_classify(t_ss, bif_ss, gnb, le,
                                          window_s=w, step_s=2.0,
                                          rtt_s=rtt_s)
        stats = analyse_convergence(results, true_label)
        conv_times.append(stats['convergence_time'])
        accuracies.append(stats['overall_accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    color = COLORS.get(true_label, 'grey')

    # Convergence time vs window size
    valid = [(w, ct) for w, ct in zip(windows, conv_times) if ct is not None]
    if valid:
        ws, cts = zip(*valid)
        ax1.plot(ws, cts, 'o-', color=color, lw=2, ms=8)
    ax1.set_xlabel("Window size (s)", fontsize=10)
    ax1.set_ylabel("Convergence time (s)", fontsize=10)
    ax1.set_title(f"Window size vs convergence time\n({true_label.upper()})",
                  fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy vs window size
    ax2.plot(windows, [a * 100 for a in accuracies], 's-',
             color=color, lw=2, ms=8)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("Window size (s)", fontsize=10)
    ax2.set_ylabel("Window-level accuracy (%)", fontsize=10)
    ax2.set_title(f"Window size vs accuracy\n({true_label.upper()})",
                  fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Window Size Sensitivity Analysis", fontsize=12)
    plt.tight_layout()

    path = os.path.join(out_dir,
                        f"stream_window_sensitivity_{true_label}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return dict(zip(windows, zip(conv_times, accuracies)))


# ══════════════════════════════════════════════════════════════════════════════
# BATCH: run on all CSVs
# ══════════════════════════════════════════════════════════════════════════════

def run_all(csv_dir, gnb, le,
            window_s=DEFAULT_WINDOW_S, step_s=DEFAULT_STEP_S):
    """Run streaming classification on every CSV in csv_dir."""
    files = sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv')))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")

    all_stats = []

    for fpath in files:
        fname    = os.path.basename(fpath)
        cc_match = re.search(r'cc-(\w+)_', fname)
        if not cc_match:
            continue
        true_label = cc_match.group(1)

        print(f"\n  {fname}  (true={true_label})")

        try:
            t, bif       = compute_bif(fpath, SERVER_IP)
            t_s, bif_s   = smooth_bif(t, bif, RTT_S)
            t_ss, bif_ss = remove_slow_start(t_s, bif_s)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        results = sliding_window_classify(
            t_ss, bif_ss, gnb, le, window_s=window_s, step_s=step_s
        )

        if not results:
            print("    No windows produced.")
            continue

        # Print window-by-window summary
        print(f"    {'t_now':>6}  {'prediction':<10}  {'conf':>6}  {'segs':>4}")
        print(f"    {'─'*6}  {'─'*10}  {'─'*6}  {'─'*4}")
        for r in results:
            mark = '✓' if r['label'] == true_label else '✗'
            print(f"    {r['t_now']:>6.1f}  "
                  f"{r['label']:<10}  "
                  f"{r['confidence']:>5.0%}  "
                  f"{r['n_segments']:>4}  {mark}")

        stats = analyse_convergence(results, true_label)
        stats['file'] = fname
        all_stats.append(stats)

        print(f"\n    Convergence: {stats['convergence_time']:.1f}s"
              if stats['convergence_time']
              else "\n    Did not converge")
        print(f"    Overall accuracy: {stats['overall_accuracy']:.0%}")

        # Per-trace timeline plot
        plot_timeline(results, true_label, fname,
                      t_ss, bif_ss, OUT_DIR, window_s)

    return all_stats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    gnb = joblib.load(os.path.join(MODEL_DIR, 'gnb.pkl'))
    le  = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    return gnb, le


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nebby streaming / online CCA classification'
    )
    parser.add_argument('csv', nargs='?', default=None,
                        help='Single CSV to classify (omit to run on all CSVs)')
    parser.add_argument('--window', type=float, default=DEFAULT_WINDOW_S,
                        help=f'Window size in seconds (default {DEFAULT_WINDOW_S})')
    parser.add_argument('--step',   type=float, default=DEFAULT_STEP_S,
                        help=f'Step size in seconds   (default {DEFAULT_STEP_S})')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run window-size sensitivity analysis on the given CSV')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Nebby — Streaming Classification")
    print(f"  window={args.window}s   step={args.step}s")
    print("=" * 60)

    gnb, le = load_model()

    # ── single file mode ──────────────────────────────────────────────────
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"File not found: {args.csv}")
            sys.exit(1)

        fname      = os.path.basename(args.csv)
        cc_match   = re.search(r'cc-(\w+)_', fname)
        true_label = cc_match.group(1) if cc_match else 'unknown'

        print(f"\nProcessing: {fname}  (true={true_label})\n")

        t, bif       = compute_bif(args.csv, SERVER_IP)
        t_s, bif_s   = smooth_bif(t, bif, RTT_S)
        t_ss, bif_ss = remove_slow_start(t_s, bif_s)

        if args.sensitivity:
            print("Running window-size sensitivity analysis...")
            sens = plot_window_size_sensitivity(
                args.csv, gnb, le, OUT_DIR
            )
            print("\nWindow sensitivity results:")
            print(f"  {'Window':>8}  {'Conv time':>10}  {'Accuracy':>10}")
            for w, (ct, acc) in sens.items():
                ct_str = f"{ct:.1f}s" if ct else "never"
                print(f"  {w:>6}s    {ct_str:>10}  {acc:>9.0%}")
        else:
            results = sliding_window_classify(
                t_ss, bif_ss, gnb, le,
                window_s=args.window, step_s=args.step
            )
            stats = analyse_convergence(results, true_label)
            plot_timeline(results, true_label, fname,
                          t_ss, bif_ss, OUT_DIR, args.window)

            print(f"\nResult:")
            print(f"  Most common prediction : {stats['most_common_pred']}")
            print(f"  Overall accuracy       : {stats['overall_accuracy']:.0%}")
            if stats['convergence_time']:
                print(f"  Converged at           : {stats['convergence_time']:.1f}s")
                print(f"  Confidence at conv.    : {stats['convergence_conf']:.0%}")
            else:
                print("  Did not converge to stable correct answer")

    # ── batch mode: run on all CSVs ───────────────────────────────────────
    else:
        all_stats = run_all(CSV_DIR, gnb, le,
                            window_s=args.window, step_s=args.step)

        # Save summary CSV
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(OUT_DIR, 'stream_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")

        # Summary convergence plot
        plot_all_convergence(all_stats, OUT_DIR)

        # Window sensitivity on first file of each CCA
        seen_cca = set()
        files    = sorted(glob.glob(os.path.join(CSV_DIR, '*_tcp.csv')))
        for fpath in files:
            cc_match = re.search(r'cc-(\w+)_', os.path.basename(fpath))
            if not cc_match:
                continue
            cca = cc_match.group(1)
            if cca not in seen_cca:
                seen_cca.add(cca)
                print(f"\nWindow sensitivity for {cca}...")
                plot_window_size_sensitivity(fpath, gnb, le, OUT_DIR)

        print(f"\nAll outputs in  {OUT_DIR}/")