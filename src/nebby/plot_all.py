"""
plot_all.py — Plot BiF traces for every CCA in separate subplots
Handles both loss-based CCAs (sawtooth) and BBR (probe pattern).

Usage:
    python3 plot_all.py                        # uses ../candidates-measurements/
    python3 plot_all.py --dir /path/to/csvs
    python3 plot_all.py --window 60            # show only first 60 seconds

Outputs (saved to ../evaluation/):
    bif_all_ccas.png         — one subplot per CCA (all traces overlaid per CCA)
    bif_per_delay.png        — side-by-side 50ms vs 100ms for each CCA
    bif_bbr_detail.png       — zoomed BBR plot showing ProbeBW + ProbeRTT
    bif_derivative.png       — d(BiF)/dt for each CCA (helps tune BBR detector)
"""

import os, sys, glob, re, argparse
from train import pair_traces
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR   = '../candidates-measurements'
OUT_DIR   = '../evaluation'
SERVER_IP = '10.0.0.1'
RTT_S     = 0.1      # seconds (2 × 50ms one-way delay)

# Colour per CCA — extended palette for 17 CCAs
CCA_COLORS = {
    'bbr'      : '#e63946',   # red
    'cubic'    : '#2196F3',   # blue
    'reno'     : '#FF9800',   # orange
    'bic'      : '#9C27B0',   # purple
    'htcp'     : '#00BCD4',   # cyan
    'illinois' : '#4CAF50',   # green
    'westwood' : '#795548',   # brown
    'yeah'     : '#607D8B',   # blue-grey
    'vegas'    : '#E91E63',   # pink
    'veno'     : '#FF5722',   # deep orange
    'scalable' : '#009688',   # teal
    'highspeed': '#FFC107',   # amber
    'hybla'    : '#3F51B5',   # indigo
    'cdg'      : '#8BC34A',   # light green
    'dctcp'    : '#F44336',   # red variant
    'lp'       : '#673AB7',   # deep purple
    'nv'       : '#FFEB3B',   # yellow
}

LOSS_BASED = {'cubic', 'reno', 'bic', 'htcp', 'illinois', 'westwood',
              'yeah', 'veno', 'scalable', 'highspeed', 'hybla',
              'cdg', 'dctcp', 'lp', 'nv', 'vegas', 'westwood'}
RATE_BASED = {'bbr'}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_trace(fpath, server_ip=SERVER_IP, rtt_s=RTT_S):
    """Load, compute BiF, smooth. Returns (t, bif_raw, bif_smooth, meta)."""
    fname    = os.path.basename(fpath)
    cc_match = re.search(r'cc-(\w+)_', fname)
    d_match  = re.search(r'bw-(\d+)_', fname)
    bw_match = re.search(r'buf-(\d+)_', fname)

    meta = {
        'cc'   : cc_match.group(1)  if cc_match  else 'unknown',
        'bw'   : int(d_match.group(1))   if d_match   else 0,
        'buf'  : int(bw_match.group(1))  if bw_match  else 0,
        'fname': fname,
        'fpath': fpath,
    }

    # detect delay from filename (delay50 or delay100 style, or from run params)
    delay_match = re.search(r'_(\d+)_\d+_tcp', fname)
    # fallback: infer RTT from filename timestamp pattern — use default
    meta['rtt_s'] = rtt_s

    t, bif_raw     = compute_bif(fpath, server_ip)
    t_s, bif_s     = smooth_bif(t, bif_raw, rtt_s)
    return t_s, bif_raw, bif_s, meta


def group_by_cca(csv_dir):
    """Return dict: {cca_name: [fpath, ...]}"""
    groups = {}
    for fpath in sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv'))):
        fname    = os.path.basename(fpath)
        cc_match = re.search(r'cc-(\w+)_', fname)
        if not cc_match:
            continue
        cc = cc_match.group(1)
        groups.setdefault(cc, []).append(fpath)
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — one subplot per CCA, all traces for that CCA overlaid
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_ccas(groups, out_dir, window_s=None):
    """
    One subplot per CCA.
    Multiple traces for the same CCA are overlaid (different alpha).
    Loss-based CCAs show raw + smoothed BiF.
    BBR gets a dedicated zoomed view (see plot_bbr_detail).
    """
    ccas = sorted(groups.keys())
    n    = len(ccas)
    if n == 0:
        print("No traces found.")
        return

    cols = 3
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(7 * cols, 4 * rows))
    gs   = gridspec.GridSpec(rows, cols, figure=fig,
                             hspace=0.55, wspace=0.35)

    for idx, cca in enumerate(ccas):
        ax    = fig.add_subplot(gs[idx // cols, idx % cols])
        color = CCA_COLORS.get(cca, '#555555')
        files = groups[cca]

        loaded = []
        for fpath in files:
            try:
                t, bif_raw, bif_s, meta = load_trace(fpath)
                loaded.append((t, bif_raw, bif_s))
            except Exception as e:
                print(f"  SKIP {os.path.basename(fpath)}: {e}")

        if not loaded:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9)
            ax.set_title(cca.upper(), fontsize=10)
            continue

        for i, (t, bif_raw, bif_s) in enumerate(loaded):
            # Clip to window
            if window_s:
                mask   = t <= (t[0] + window_s)
                t      = t[mask]
                bif_raw = bif_raw[mask]
                bif_s   = bif_s[mask]

            alpha_raw  = max(0.1, 0.4 / len(loaded))
            alpha_smooth = max(0.5, 0.9 / len(loaded))

            ax.fill_between(t, 0, bif_raw / 1024,
                            alpha=alpha_raw * 0.5, color=color)
            ax.plot(t, bif_raw / 1024,
                    color=color, lw=0.5, alpha=alpha_raw)
            ax.plot(t, bif_s   / 1024,
                    color=color, lw=1.8, alpha=alpha_smooth,
                    label='smoothed' if i == 0 else None)

        ax.set_title(
            f"{cca.upper()}  "
            f"({'rate-based' if cca in RATE_BASED else 'loss-based'})\n"
            f"{len(loaded)} trace(s)",
            fontsize=9,
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        # Annotate expected shape
        if cca == 'bbr':
            ax.text(0.02, 0.95,
                    'Look for: flat + periodic bumps (ProbeBW)',
                    transform=ax.transAxes, fontsize=6,
                    va='top', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='lightyellow', alpha=0.8))
        elif cca in {'cubic', 'bic', 'htcp'}:
            ax.text(0.02, 0.95,
                    'Look for: cubic sawtooth',
                    transform=ax.transAxes, fontsize=6,
                    va='top', color='navy',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='lightyellow', alpha=0.8))
        elif cca in {'reno', 'westwood', 'hybla'}:
            ax.text(0.02, 0.95,
                    'Look for: linear sawtooth',
                    transform=ax.transAxes, fontsize=6,
                    va='top', color='navy',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='lightyellow', alpha=0.8))
        elif cca in {'vegas', 'veno', 'lp', 'nv'}:
            ax.text(0.02, 0.95,
                    'Look for: smooth / delay-based control',
                    transform=ax.transAxes, fontsize=6,
                    va='top', color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='lightyellow', alpha=0.8))

    # Hide unused subplots
    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    fig.suptitle(
        "BiF Traces — All CCAs\n"
        "(smoothed = thick line, raw = thin fill)",
        fontsize=13, y=1.01,
    )

    path = os.path.join(out_dir, 'bif_all_ccas.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — BBR detail: zoom in to show ProbeBW and ProbeRTT
# ══════════════════════════════════════════════════════════════════════════════

def plot_bbr_detail(groups, out_dir, rtt_s=RTT_S):
    """
    Dedicated BBR plot with annotations showing:
      - ProbeBW spikes (every 8 RTTs)
      - ProbeRTT dips (every 10s for BBRv1)
      - First derivative to help tune detect_bbr()
    """
    if 'bbr' not in groups or not groups['bbr']:
        print("No BBR traces found — skipping BBR detail plot.")
        return

    fpath = groups['bbr'][0]   # use first BBR trace

    try:
        t, bif_raw, bif_s, meta = load_trace(fpath, rtt_s=rtt_s)
    except Exception as e:
        print(f"BBR detail: {e}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── show 3 time windows ───────────────────────────────────────────────
    # Full trace overview
    ax_full = axes[0]
    ax_full.plot(t, bif_raw / 1024, color='lightcoral',
                 lw=0.6, alpha=0.5, label='Raw BiF')
    ax_full.plot(t, bif_s   / 1024, color='#e63946',
                 lw=2.0, label='Smoothed BiF')
    ax_full.set_ylabel("KB in flight", fontsize=9)
    ax_full.set_title("BBR — Full Trace Overview", fontsize=10)
    ax_full.legend(fontsize=8)
    ax_full.grid(True, alpha=0.25)

    # Zoomed: 20-40s (should show ProbeBW pattern)
    ax_zoom = axes[1]
    mask    = (t >= 20) & (t <= 40)
    ax_zoom.plot(t[mask], bif_raw[mask] / 1024,
                 color='lightcoral', lw=0.8, alpha=0.5, label='Raw BiF')
    ax_zoom.plot(t[mask], bif_s[mask]   / 1024,
                 color='#e63946', lw=2.0, label='Smoothed BiF')

    # Annotate expected ProbeBW period
    probe_bw_period = 8 * rtt_s
    ax_zoom.set_ylabel("KB in flight", fontsize=9)
    ax_zoom.set_title(
        f"BBR — Zoomed 20–40s  "
        f"(ProbeBW expected every {probe_bw_period:.1f}s,  "
        f"ProbeRTT every ~10s)",
        fontsize=10,
    )
    # Draw expected ProbeBW lines
    for probe_t in np.arange(20, 40, probe_bw_period):
        ax_zoom.axvline(probe_t, color='blue', lw=0.8,
                        alpha=0.4, linestyle='--')
    ax_zoom.legend(fontsize=8)
    ax_zoom.grid(True, alpha=0.25)

    # First derivative (used by detect_bbr)
    ax_deriv = axes[2]
    if len(t) > 1:
        dbif = np.diff(bif_s) / np.diff(t)
        t_d  = t[1:]
        p95  = np.percentile(dbif, 95)
        p05  = np.percentile(dbif, 5)

        ax_deriv.plot(t_d, dbif / 1024, color='purple',
                      lw=0.8, alpha=0.7, label='d(BiF)/dt')
        ax_deriv.axhline(p95 / 1024, color='blue',  lw=1.2,
                         linestyle='--', label=f'p95 (spike threshold)')
        ax_deriv.axhline(p05 / 1024, color='red',   lw=1.2,
                         linestyle='--', label=f'p05 (dip threshold)')
        ax_deriv.axhline(0,           color='black', lw=0.6, alpha=0.5)

        # Highlight detected spikes and dips
        spike_mask = dbif > p95
        dip_mask   = dbif < p05
        ax_deriv.scatter(t_d[spike_mask], dbif[spike_mask] / 1024,
                         color='blue', s=15, zorder=4, label='ProbeBW spikes')
        ax_deriv.scatter(t_d[dip_mask],   dbif[dip_mask]   / 1024,
                         color='red',  s=15, zorder=4, label='ProbeRTT dips')

    ax_deriv.set_xlabel("Time (s)", fontsize=9)
    ax_deriv.set_ylabel("d(BiF)/dt  (KB/s²)", fontsize=9)
    ax_deriv.set_title(
        "BBR — First Derivative  "
        "(used by rule-based BBR detector in classify.py)",
        fontsize=10,
    )
    ax_deriv.legend(fontsize=7, loc='upper right')
    ax_deriv.grid(True, alpha=0.25)

    fig.suptitle("BBR Detail Analysis", fontsize=13)
    plt.tight_layout()

    path = os.path.join(out_dir, 'bif_bbr_detail.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Side-by-side: 50ms delay vs 100ms delay per CCA
#          (replicates paper Figure 4 structure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_delay(csv_dir, out_dir, window_s=40):
    """
    Final correct version:
    - Uses SAME pairing logic as training (pair_traces)
    - No ordering assumptions
    - Guarantees correct 50ms vs 100ms pairing
    """

    pairs, unpaired = pair_traces(csv_dir)

    if not pairs:
        print("No valid pairs found.")
        return

    if unpaired:
        print(f"WARNING: {len(unpaired)} unpaired file(s) skipped")

    # Group pairs by CCA
    by_cca = {}
    for cca, f50, f100 in pairs:
        by_cca.setdefault(cca, []).append((f50, f100))

    ccas = sorted(by_cca.keys())
    n    = len(ccas)

    fig, axes_grid = plt.subplots(
        n, 2,
        figsize=(14, 3.5 * n),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for row, cca in enumerate(ccas):
        color = CCA_COLORS.get(cca, '#555555')
        pairs_list = by_cca[cca]

        ax_left  = axes_grid[row][0]   # 50ms
        ax_right = axes_grid[row][1]   # 100ms

        for f50, f100 in pairs_list:
            try:
                # IMPORTANT: correct RTT for smoothing
                t1, _, bif1, _ = load_trace(f50, rtt_s=0.1)
                t2, _, bif2, _ = load_trace(f100, rtt_s=0.2)
            except Exception as e:
                print(f"SKIP {f50}, {f100}: {e}")
                continue

            # Apply time window
            m1 = t1 <= (t1[0] + window_s)
            m2 = t2 <= (t2[0] + window_s)

            t1, bif1 = t1[m1], bif1[m1]
            t2, bif2 = t2[m2], bif2[m2]

            # Plot 50ms
            ax_left.plot(t1, bif1 / 1024,
                         color=color, lw=1.5, alpha=0.8)
            ax_left.fill_between(t1, 0, bif1 / 1024,
                                 color=color, alpha=0.15)

            # Plot 100ms
            ax_right.plot(t2, bif2 / 1024,
                          color=color, lw=1.5, alpha=0.8)
            ax_right.fill_between(t2, 0, bif2 / 1024,
                                  color=color, alpha=0.15)

        ax_left.set_title(f"{cca.upper()} — 50ms RTT", fontsize=9)
        ax_right.set_title(f"{cca.upper()} — 100ms RTT", fontsize=9)

        for ax in (ax_left, ax_right):
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("KB in flight", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

    fig.suptitle(
        "BiF per CCA per Delay (Correct Pairing)",
        fontsize=13, y=1.01,
    )

    path = os.path.join(out_dir, 'bif_per_delay.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Derivative plot for all CCAs (helps debug segmentation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_derivatives(groups, out_dir, window_s=60):
    """
    Plot d(BiF)/dt for each CCA.
    - Loss-based CCAs: derivative spikes at loss events (back-offs)
    - BBR: derivative spikes at ProbeBW, dips at ProbeRTT
    Useful for tuning drop_fraction in segment_bif() and detect_bbr().
    """
    ccas = sorted(groups.keys())
    n    = len(ccas)
    cols = 3
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(7 * cols, 3.5 * rows))
    gs  = gridspec.GridSpec(rows, cols, figure=fig,
                            hspace=0.55, wspace=0.35)

    for idx, cca in enumerate(ccas):
        ax    = fig.add_subplot(gs[idx // cols, idx % cols])
        color = CCA_COLORS.get(cca, '#555555')
        files = groups[cca]

        plotted = False
        for fpath in files[:1]:   # just first trace per CCA
            try:
                t, _, bif_s, _ = load_trace(fpath)
            except Exception:
                continue

            mask = t <= (t[0] + window_s)
            t_w  = t[mask]
            bs_w = bif_s[mask]

            if len(t_w) > 2:
                dbif = np.diff(bs_w) / np.diff(t_w)
                t_d  = t_w[1:]

                # Clip extreme outliers for visibility
                p1, p99 = np.percentile(dbif, 1), np.percentile(dbif, 99)
                dbif_clipped = np.clip(dbif, p1, p99)

                ax.plot(t_d, dbif_clipped / 1024,
                        color=color, lw=0.8, alpha=0.8)
                ax.axhline(0, color='black', lw=0.6, alpha=0.5)
                ax.fill_between(t_d,
                                np.where(dbif_clipped < 0,
                                         dbif_clipped / 1024, 0),
                                0, alpha=0.3, color='red',
                                label='back-off regions')
                plotted = True

        if not plotted:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', fontsize=9)

        ax.set_title(f"{cca.upper()} — d(BiF)/dt", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB/s²", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6)

    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    fig.suptitle(
        "First Derivative of BiF — All CCAs\n"
        "(negative spikes = back-offs used for segmentation)",
        fontsize=13, y=1.01,
    )

    path = os.path.join(out_dir, 'bif_derivative.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot BiF traces for all CCAs'
    )
    parser.add_argument('--dir',    default=CSV_DIR,
                        help=f'CSV directory (default: {CSV_DIR})')
    parser.add_argument('--window', type=float, default=None,
                        help='Show only first N seconds (default: full trace)')
    parser.add_argument('--rtt',    type=float, default=RTT_S,
                        help=f'RTT in seconds for smoothing (default: {RTT_S})')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Nebby — Plot All CCAs")
    print(f"  CSV dir : {args.dir}")
    print(f"  Window  : {args.window or 'full trace'}")
    print("=" * 60 + "\n")

    groups = group_by_cca(args.dir)

    if not groups:
        print(f"No *_tcp.csv files found in {args.dir}")
        sys.exit(1)

    print(f"Found CCAs: {sorted(groups.keys())}\n")
    for cca, files in sorted(groups.items()):
        print(f"  {cca:<12}: {len(files)} trace(s)")

    print("\nGenerating plots ...")

    plot_all_ccas(groups, OUT_DIR, window_s=args.window)
    plot_bbr_detail(groups, OUT_DIR, rtt_s=args.rtt)
    plot_per_delay(args.dir, OUT_DIR, window_s=args.window or 60)
    plot_derivatives(groups, OUT_DIR, window_s=args.window or 60)

    print(f"\nDone. All plots saved to  {OUT_DIR}/")