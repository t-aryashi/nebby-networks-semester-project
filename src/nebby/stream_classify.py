"""
stream_classify.py — Online / Streaming CCA Classification
Novel contribution beyond Nebby (SIGCOMM '24)

CHANGES:
  - Updated to load 'gnb_stream.pkl' (3D features) to prevent 6D/3D crashes.
  - Patched BBR rule-based detector invocation to prevent over-triggering
    on short windows at high bandwidth (2000 Kbps).
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

DEFAULT_WINDOW_S = 15.0   
DEFAULT_STEP_S   =  3.0   

COLORS = {'cubic': 'steelblue', 'reno': 'tomato',
          'bbr': 'seagreen',    'unknown': 'grey'}


# ══════════════════════════════════════════════════════════════════════════════
# CORE: sliding window predictor
# ══════════════════════════════════════════════════════════════════════════════

def sliding_window_classify(t, bif, gnb, le,
                             window_s=DEFAULT_WINDOW_S,
                             step_s=DEFAULT_STEP_S,
                             rtt_s=RTT_S):
    results  = []
    t_start  = t[0]
    t_end    = t[-1]

    pos = t_start + window_s

    while pos <= t_end + step_s:
        pos = min(pos, t_end)

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
    # Need at least 5 seconds of data to do anything meaningful
    if len(t_win) < 10 or (t_win[-1] - t_win[0]) < 5.0:
        return {'label': 'unknown', 'confidence': 0.0,
                'n_segments': 0, 'method': 'unknown'}

    # 1. BBR rule-based check on this window
    # PATCH: We only run the BBR check if the window has enough data to accurately calculate variance
    if (t_win[-1] - t_win[0]) >= 8.0:
        bbr = detect_bbr(t_win, bif_win, rtt_s)
        if bbr is not None:
            return {'label': 'bbr', 'confidence': 1.0,
                    'n_segments': 0, 'method': 'bbr_rule'}

    # 2. Segment and classify with GNB
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
    labels = [r['label'] for r in results]
    times  = [r['t_now'] for r in results]
    confs  = [r['confidence'] for r in results]

    correct     = [l == true_label for l in labels]
    accuracy    = sum(correct) / len(correct) if correct else 0.0

    conv_time   = None
    conv_idx    = None
    for i in range(len(labels) - stable_window + 1):
        window_labels = labels[i:i + stable_window]
        if all(l == true_label for l in window_labels):
            conv_time = times[i]
            conv_idx  = i
            break

    conv_conf = confs[conv_idx] if conv_idx is not None else None

    # Handle case where results are predominantly 'unknown'
    if not labels:
        most_common = 'unknown'
    else:
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

def plot_timeline(results, true_label, fname, bif_t, bif_vals, out_dir, window_s):
    times  = np.array([r['t_now']      for r in results])
    labels = [r['label']               for r in results]
    confs  = np.array([r['confidence'] for r in results])

    fig, (ax_bif, ax_conf) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True,
    )

    ax_bif.plot(bif_t, bif_vals / 1024,
                color='black', lw=0.8, alpha=0.4, label='BiF (smoothed)')

    for i, (t_now, label) in enumerate(zip(times, labels)):
        t_left  = t_now - window_s
        color   = COLORS.get(label, 'grey')
        correct = (label == true_label)
        alpha   = 0.25 if correct else 0.10
        ax_bif.axvspan(t_left, t_now, alpha=alpha, color=color, linewidth=0)

    patches = [mpatches.Patch(color=COLORS.get(c, 'grey'), alpha=0.5, label=c)
               for c in ['cubic', 'reno', 'bbr', 'unknown']]
    
    # PATCH: Fixed Matplotlib Warning by ensuring patch properties are explicit
    patches.append(mpatches.Patch(facecolor='white', edgecolor='black', label=f'true={true_label}'))
    
    ax_bif.legend(handles=patches, fontsize=8, loc='upper right')
    ax_bif.set_ylabel("KB in flight", fontsize=9)
    ax_bif.set_title(f"Streaming Classification — {fname}\nTrue CCA: {true_label.upper()}   Window: {window_s}s", fontsize=10)
    ax_bif.grid(True, alpha=0.25)

    for i in range(len(labels) - 2):
        if (labels[i] == true_label and labels[i+1] == true_label and labels[i+2] == true_label):
            ax_bif.axvline(times[i], color='black', lw=1.5, linestyle='--', alpha=0.7, label='converged')
            ax_bif.text(times[i] + 0.5, ax_bif.get_ylim()[1] * 0.9, f'converged\nt={times[i]:.0f}s', fontsize=7, color='black')
            break

    point_colors = [COLORS.get(l, 'grey') for l in labels]

    ax_conf.plot(times, confs, color='black', lw=1.0, alpha=0.4)
    ax_conf.scatter(times, confs, c=point_colors, s=30, zorder=3, label='confidence (colour = prediction)')

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
    df = pd.DataFrame(all_stats)
    df = df.dropna(subset=['convergence_time'])
    if df.empty:
        print("  No convergence data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for cca in sorted(df['true_label'].unique()):
        sub = df[df['true_label'] == cca]['convergence_time']
        ax.bar(cca, sub.mean(), color=COLORS.get(cca, 'grey'), alpha=0.8, yerr=sub.std() if len(sub) > 1 else 0, capsize=5, label=cca)
        for val in sub.values: ax.scatter(cca, val, color=COLORS.get(cca, 'grey'), edgecolors='black', s=50, zorder=3)
    ax.set_ylabel("Convergence time (s)", fontsize=10)
    ax.set_title("Time to stable correct prediction", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    for cca in sorted(df['true_label'].unique()):
        sub = df[df['true_label'] == cca]['overall_accuracy']
        ax2.bar(cca, sub.mean() * 100, color=COLORS.get(cca, 'grey'), alpha=0.8, yerr=sub.std() * 100 if len(sub) > 1 else 0, capsize=5)
        for val in sub.values: ax2.scatter(cca, val * 100, color=COLORS.get(cca, 'grey'), edgecolors='black', s=50, zorder=3)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("% windows correctly classified", fontsize=10)
    ax2.set_title("Window-level accuracy", fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'stream_convergence_summary.png'), dpi=150)
    plt.close()

def plot_window_size_sensitivity(csv_path, gnb, le, out_dir, server_ip=SERVER_IP, rtt_s=RTT_S):
    fname      = os.path.basename(csv_path)
    cc_match   = re.search(r'cc-(\w+)_', fname)
    true_label = cc_match.group(1) if cc_match else 'unknown'

    t, bif       = compute_bif(csv_path, server_ip)
    t_s, bif_s   = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss = remove_slow_start(t_s, bif_s)

    windows      = [5, 8, 10, 15, 20, 30]
    conv_times, accuracies = [], []

    for w in windows:
        results = sliding_window_classify(t_ss, bif_ss, gnb, le, window_s=w, step_s=2.0, rtt_s=rtt_s)
        stats = analyse_convergence(results, true_label)
        conv_times.append(stats['convergence_time'])
        accuracies.append(stats['overall_accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    color = COLORS.get(true_label, 'grey')

    valid = [(w, ct) for w, ct in zip(windows, conv_times) if ct is not None]
    if valid:
        ws, cts = zip(*valid)
        ax1.plot(ws, cts, 'o-', color=color, lw=2, ms=8)
    ax1.set_xlabel("Window size (s)", fontsize=10)
    ax1.set_ylabel("Convergence time (s)", fontsize=10)
    ax1.set_title(f"Window size vs convergence time\n({true_label.upper()})", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(windows, [a * 100 for a in accuracies], 's-', color=color, lw=2, ms=8)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("Window size (s)", fontsize=10)
    ax2.set_ylabel("Window-level accuracy (%)", fontsize=10)
    ax2.set_title(f"Window size vs accuracy\n({true_label.upper()})", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"stream_window_sensitivity_{true_label}.png"), dpi=150)
    plt.close()
    return dict(zip(windows, zip(conv_times, accuracies)))

def run_all(csv_dir, gnb, le, window_s=DEFAULT_WINDOW_S, step_s=DEFAULT_STEP_S):
    files = sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv')))
    all_stats = []

    for fpath in files:
        fname = os.path.basename(fpath)
        cc_match = re.search(r'cc-(\w+)_', fname)
        if not cc_match: continue
        true_label = cc_match.group(1)

        print(f"\n  {fname}  (true={true_label})")
        try:
            t, bif       = compute_bif(fpath, SERVER_IP)
            t_s, bif_s   = smooth_bif(t, bif, RTT_S)
            t_ss, bif_ss = remove_slow_start(t_s, bif_s)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        results = sliding_window_classify(t_ss, bif_ss, gnb, le, window_s=window_s, step_s=step_s)
        if not results:
            print("    No windows produced.")
            continue

        stats = analyse_convergence(results, true_label)
        stats['file'] = fname
        all_stats.append(stats)

        plot_timeline(results, true_label, fname, t_ss, bif_ss, OUT_DIR, window_s)

    return all_stats

def load_model():
    # PATCH: Loading the 3D Single-Profile model specifically generated for streaming
    gnb = joblib.load(os.path.join(MODEL_DIR, 'gnb_stream.pkl'))
    le  = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_stream.pkl'))
    return gnb, le

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nebby streaming / online CCA classification')
    parser.add_argument('csv', nargs='?', default=None)
    parser.add_argument('--window', type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument('--step',   type=float, default=DEFAULT_STEP_S)
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Nebby — Streaming Classification")
    print(f"  window={args.window}s   step={args.step}s")
    print("=" * 60)

    gnb, le = load_model()

    if args.csv:
        fname      = os.path.basename(args.csv)
        cc_match   = re.search(r'cc-(\w+)_', fname)
        true_label = cc_match.group(1) if cc_match else 'unknown'

        print(f"\nProcessing: {fname}  (true={true_label})\n")

        t, bif       = compute_bif(args.csv, SERVER_IP)
        t_s, bif_s   = smooth_bif(t, bif, RTT_S)
        t_ss, bif_ss = remove_slow_start(t_s, bif_s)

        if args.sensitivity:
            plot_window_size_sensitivity(args.csv, gnb, le, OUT_DIR)
        else:
            results = sliding_window_classify(t_ss, bif_ss, gnb, le, window_s=args.window, step_s=args.step)
            stats = analyse_convergence(results, true_label)
            plot_timeline(results, true_label, fname, t_ss, bif_ss, OUT_DIR, args.window)

            print(f"\nResult:\n  Most common prediction : {stats['most_common_pred']}\n  Overall accuracy       : {stats['overall_accuracy']:.0%}")
            if stats['convergence_time']:
                print(f"  Converged at           : {stats['convergence_time']:.1f}s")
    else:
        all_stats = run_all(CSV_DIR, gnb, le, window_s=args.window, step_s=args.step)
        pd.DataFrame(all_stats).to_csv(os.path.join(OUT_DIR, 'stream_summary.csv'), index=False)
        plot_all_convergence(all_stats, OUT_DIR)