"""
selenium_measure.py — Sequential multi-asset CCA measurement
Paper reference: Nebby §3.5 and §4.5

WHY SEQUENTIAL INSTEAD OF PARALLEL:
  Parallel flows at 400 Kbps each cause two fatal problems:
    1. BBR detection fails — ProbeRTT dips bring BiF to near 0, making
       coefficient of variation too high (CV > 0.35 threshold).
    2. Loss-based CCAs never exit slow start — at 400 Kbps, a 1MB file
       transfers in 20s but the tiny shared buffer causes the first loss
       so late that the entire transfer is essentially slow start.

  Sequential flows each get the full 2000 Kbps:
    - BBR shows clean flat cruise + small periodic probes → detected
    - CUBIC/Reno reach congestion avoidance within 5-8s → classifiable
    - Each flow gets its own pcap → clean separation

  Paper §3.5: "we ran a modified version of Nebby with our Selenium
  client that creates a separate bottleneck queue to isolate each
  connection so that each flow can be classified separately."
  → Sequential = one bottleneck queue per flow = paper's approach.

PURPOSE (paper Table 8):
  Same webpage → different CCAs per asset type:
    video  → BBR    (video CDN)
    images → CUBIC  (image CDN)
    CSS/JS → Reno   (static CDN)

  Server assigns CCA per connection via TCP_CONGESTION socket option.
  NOVEL: We know the true CCA → can verify accuracy (paper could not).

Usage:
    # Terminal 1 — start server (must be running):
    sudo python3 selenium_server.py

    # Terminal 2:
    python3 selenium_measure.py
    python3 selenium_measure.py --bw 2000 --delay 50

Outputs (../evaluation/selenium/):
    selenium_flows.png
    selenium_summary.txt
"""

import os, sys, re, argparse, subprocess, tempfile, shutil, time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
for candidate in [_this_dir,
                  os.path.join(_this_dir, '../nebby'),
                  os.path.join(_this_dir, '../../nebby')]:
    if os.path.exists(os.path.join(candidate, 'bif.py')):
        sys.path.insert(0, candidate)
        break

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features
from classify   import detect_bbr, load_model, _majority_vote

# ── config ────────────────────────────────────────────────────────────────────
SERVER_IP   = '10.0.0.1'
SERVER_PORT = 80
BASE_URL    = f'http://{SERVER_IP}:{SERVER_PORT}'

DEFAULT_BW    = 2000   # Kbps — matches training data
DEFAULT_DELAY = 50     # ms one-way

TRACES_DIR = '../traces'
OUT_DIR    = '../evaluation/selenium'
MODEL_DIR  = '../models'

# Assets — served sequentially, each gets full BW
# File sizes tuned for 2000 Kbps: needs 15+ seconds of transfer
# At 2000 Kbps: 2MB = 8s (enough), 4MB = 16s (better for BBR)
ASSETS = [
    {'file': 'video.bin',  'label': 'video',         'true_cca': 'bbr',   'size_mb': 4},
    {'file': 'image1.bin', 'label': 'image (BBR)',   'true_cca': 'bbr',   'size_mb': 2},
    {'file': 'image2.bin', 'label': 'image (CUBIC)', 'true_cca': 'cubic', 'size_mb': 2},
    {'file': 'style.css',  'label': 'CSS (CUBIC)',   'true_cca': 'cubic', 'size_mb': 2},
    {'file': 'script.js',  'label': 'JS (Reno)',     'true_cca': 'reno',  'size_mb': 2},
]

# At 2000 Kbps: 4MB = 16s, 2MB = 8s
# wget timeout must exceed longest file transfer
WGET_TIMEOUT   = 60    # seconds per flow
SUBPROCESS_TIMEOUT = 90  # includes tcpdump start/stop overhead

MIN_FLOW_BYTES = 500 * 1024  # 500KB — flows smaller than this are skipped

CCA_COLORS = {
    'bbr':     '#e63946',
    'cubic':   '#2196F3',
    'reno':    '#FF9800',
    'bic':     '#9C27B0',
    'unknown': '#aaaaaa',
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Bandwidth trace
# ══════════════════════════════════════════════════════════════════════════════

def generate_bw_trace(bw_kbps, traces_dir=TRACES_DIR, duration_s=120):
    os.makedirs(traces_dir, exist_ok=True)
    path = os.path.join(traces_dir, f'bw_{bw_kbps}.trace')
    if os.path.exists(path):
        return path
    bw_bytes = bw_kbps * 1000 // 8
    pps = max(1, bw_bytes // 1500)
    with open(path, 'w') as f:
        last_t = 0
        for t in range(duration_s):
            for i in range(pps):
                ct = t * 1000 + i * 1000 // pps
                ct = max(ct, last_t + 1)
                f.write(f"{ct}\n")
                last_t = ct
    print(f"  Generated trace: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Capture ONE asset flow via Mahimahi
# ══════════════════════════════════════════════════════════════════════════════

def capture_one_asset(asset, pcap_path, bw, delay, trace_path):
    """
    Capture a single asset download through Mahimahi.

    Architecture mirrors btl.sh (the proven working capture):

        mm-delay [delay]
            tcpdump -i ingress   ← capture here, in mm-delay shell
            mm-link [bw trace]
                wget [url]       ← download here, inside mm-link

    tcpdump runs in the mm-delay network namespace on the ingress
    interface. wget runs inside mm-link. This is exactly how
    btl.sh captures traffic in the dataset generation pipeline.
    """
    url = f'{BASE_URL}/{asset["file"]}'

    # wget command that runs INSIDE mm-link
    wget_cmd = (
        f'wget --tries=1 --timeout={WGET_TIMEOUT} '
        f'"{url}" -O /dev/null -q 2>/dev/null'
    )

    # mm-link wraps wget — nested inside mm-delay
    mm_link_cmd = (
        f'mm-link {trace_path} {trace_path} -- '
        f'{wget_cmd}'
    )

    # Outer shell: tcpdump on ingress + mm-link nested inside
    # This mirrors btl.sh exactly
    inner_cmd = (
        f'tcpdump -i ingress -w {pcap_path} -q & '
        f'DPID=$! ; '
        f'sleep 0.3 ; '
        f'{mm_link_cmd} ; '
        f'sleep 1 ; '
        f'kill $DPID 2>/dev/null ; '
        f'wait $DPID 2>/dev/null'
    )

    mahimahi_cmd = [
        'mm-delay', str(delay),
        'bash', '-c', inner_cmd,
    ]

    result = subprocess.run(
        mahimahi_cmd,
        capture_output=True, text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    if not os.path.exists(pcap_path):
        return False, 0

    size = os.path.getsize(pcap_path)
    return size > 5000, size  # 5KB minimum — real traffic


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Convert pcap to CSV
# ══════════════════════════════════════════════════════════════════════════════

def pcap_to_csv(pcap_path, csv_path):
    """Convert a pcap to a tshark CSV with TCP fields."""
    cmd = [
        'tshark', '-r', pcap_path,
        '-Y', 'tcp',
        '-T', 'fields',
        '-e', 'frame.time_relative',
        '-e', 'frame.time_delta',
        '-e', 'ip.src',
        '-e', 'tcp.len',
        '-e', 'tcp.seq',
        '-e', 'tcp.ack',
        '-e', 'tcp.window_size',
        '-E', 'header=y',
        '-E', 'separator=,',
    ]
    with open(csv_path, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, timeout=30)

    rows = sum(1 for _ in open(csv_path)) - 1 if os.path.exists(csv_path) else 0
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify one flow's CSV
# ══════════════════════════════════════════════════════════════════════════════

def classify_flow(csv_path, gnb, le, rtt_s):
    """
    Classify a single-flow CSV.
    Each flow has the full BW (no sharing) so BiF shapes are clean.
    """
    try:
        t, bif     = compute_bif(csv_path, server_ip=None)
        t_s, bif_s = smooth_bif(t, bif, rtt_s)
    except Exception as e:
        return 'unknown', 0.0, np.array([0.0]), np.array([0.0])

    if len(t_s) < 10:
        return 'unknown', 0.0, t_s, bif_s

    # BBR check on full smoothed trace
    bbr = detect_bbr(t_s, bif_s, rtt_s)
    if bbr:
        return 'bbr', 1.0, t_s, bif_s

    # Remove slow start
    duration = t_s[-1] - t_s[0]
    if duration > 5.0:
        try:
            t_ca, bif_ca = remove_slow_start(t_s, bif_s)
        except Exception:
            t_ca, bif_ca = t_s, bif_s
    else:
        cut = max(1, len(t_s) // 8)
        t_ca, bif_ca = t_s[cut:], bif_s[cut:]

    if len(bif_ca) < 10:
        return 'unknown', 0.0, t_s, bif_s

    # Dynamic BiF floor from actual trace
    bif_floor = max(500, int(bif_ca.mean() * 0.05))

    segments = segment_bif(t_ca, bif_ca,
                            drop_fraction=0.30,
                            min_duration_s=0.3,
                            min_points=8,
                            bif_min=bif_floor)
    if len(segments) == 0:
        segments = segment_bif(t_ca, bif_ca,
                                drop_fraction=0.20,
                                min_duration_s=0.2,
                                min_points=5,
                                bif_min=200)

    feats = extract_features(segments)
    if len(feats) == 0:
        return 'unknown', 0.0, t_s, bif_s

    n_feat = gnb.theta_.shape[1]
    if n_feat == 6 and feats.shape[1] == 3:
        feats = np.hstack([feats, feats])

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)
    return label, conf, t_s, bif_s


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Plot all flows
# ══════════════════════════════════════════════════════════════════════════════

def plot_flows(results, out_dir):
    n = len(results)
    if n == 0:
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(7 * cols, 4 * rows))

    for idx, r in enumerate(results):
        ax    = fig.add_subplot(rows, cols, idx + 1)
        t     = r['t']
        bif   = r['bif']
        pred  = r['pred_cca']
        true  = r['true_cca']
        color = CCA_COLORS.get(pred, '#aaaaaa')
        bc    = 'green' if r['correct'] else 'red'

        if len(t) > 5:
            bif_roll = (pd.Series(bif)
                        .rolling(10, center=True, min_periods=1)
                        .mean().values)
            ax.fill_between(t, 0, bif / 1024, alpha=0.15, color=color)
            ax.plot(t, bif / 1024,      color=color, lw=0.6, alpha=0.4)
            ax.plot(t, bif_roll / 1024, color=color, lw=2.2)
        else:
            ax.text(0.5, 0.5, 'Too short', transform=ax.transAxes,
                    ha='center', fontsize=10)

        mark = '✓' if r['correct'] else '✗'
        ax.set_title(
            f"{r['label']}\n"
            f"True: {true.upper()}  Pred: {pred.upper()}  {mark}  "
            f"conf={r['confidence']:.0%}",
            fontsize=9, color=bc, fontweight='bold',
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2.5)

    fig.suptitle(
        "Nebby Selenium Measurement — Multi-Asset CCA Classification\n"
        "Replicates paper Table 8  |  Green border = correct prediction\n"
        "Each asset served with a different CCA (TCP_CONGESTION socket option)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(out_dir, 'selenium_flows.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Report
# ══════════════════════════════════════════════════════════════════════════════

def save_report(results, out_dir):
    classified = [r for r in results if r['pred_cca'] != 'unknown']
    correct    = [r for r in classified if r['correct']]
    acc        = len(correct) / len(classified) if classified else 0.0

    lines = [
        "Nebby — Multi-Asset Browser Measurement Report",
        "(Equivalent to paper Table 8)",
        "=" * 65,
        "",
        "Each asset fetched sequentially at full 2000 Kbps.",
        "Server assigns per-socket CCA via TCP_CONGESTION option.",
        "This simulates different CDNs serving different content types.",
        "",
        f"{'ASSET':<22} {'TRUE CCA':<10} {'PRED CCA':<12} "
        f"{'CONF':>5}  {'SIZE':>6}  OK?",
        "─" * 65,
    ]

    for r in results:
        mark = ('✓' if r['correct'] else '✗') \
               if r['pred_cca'] != 'unknown' else '─'
        lines.append(
            f"{r['label']:<22} {r['true_cca']:<10} "
            f"{r['pred_cca']:<12} {r['confidence']:>4.0%}  "
            f"{r['size_mb']:>4}MB  {mark}"
        )

    lines += [
        "",
        f"Classified : {len(classified)}/{len(results)} flows",
        f"Accuracy   : {len(correct)}/{len(classified)} = {acc:.0%}"
        if classified else "No classifiable flows",
        "",
        "Asset type → Predicted CCA  (paper Table 8 equivalent):",
    ]
    for r in results:
        if r['pred_cca'] != 'unknown':
            lines.append(f"  {r['label']:<22} → {r['pred_cca'].upper()}")

    lines += [
        "",
        "Paper finding (§4.5 Table 8):",
        "  Video/audio → BBR  |  Static assets → CUBIC/Reno",
        "  'BBR seems to be the CCA of choice for video streaming flows,",
        "   while CUBIC is often used for static content.'",
        "",
        "NOVEL CONTRIBUTION:",
        "  Paper observed real websites — could NOT verify accuracy.",
        "  We set the CCA and VERIFY predictions → ground truth numbers.",
        f"  Our accuracy: {acc:.0%}  (paper could not compute this)",
    ]

    report = '\n'.join(lines)
    print("\n" + report)

    path = os.path.join(out_dir, 'selenium_summary.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def check_server(delay):
    cmd = ['mm-delay', str(delay), 'bash', '-c',
           f'wget -q --tries=1 --timeout=5 '
           f'--spider {BASE_URL}/ 2>&1 | head -3']
    try:
        r = subprocess.run(cmd, capture_output=True,
                           text=True, timeout=15)
        combined = r.stdout + r.stderr
        return '200' in combined or 'connected' in combined.lower()
    except Exception:
        return False


def run(bw=DEFAULT_BW, delay=DEFAULT_DELAY):
    os.makedirs(OUT_DIR, exist_ok=True)
    rtt_s      = delay * 2 / 1000
    trace_path = generate_bw_trace(bw)

    print("=" * 65)
    print("  Nebby — Sequential Multi-Asset Browser Measurement")
    print(f"  Server : {BASE_URL}")
    print(f"  BW     : {bw} Kbps  Delay: {delay}ms  RTT: {rtt_s*1000:.0f}ms")
    print(f"  Assets : {len(ASSETS)} files (sequential, each gets full BW)")
    print("=" * 65)

    try:
        gnb, le = load_model()
        n_feat  = gnb.theta_.shape[1]
        print(f"  Model  : {n_feat}D  classes={list(le.classes_)}\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("  Checking server...")
    if check_server(delay):
        print("  Server OK.\n")
    else:
        print("  WARNING: Could not verify server. Proceeding anyway.\n")
        print("  If wget fails, check: sudo python3 selenium_server.py\n")

    run_dir = tempfile.mkdtemp(prefix='nebby_seq_')
    results = []

    try:
        for asset in ASSETS:
            fname    = asset['file']
            size_mb  = asset['size_mb']
            est_time = int(size_mb * 1024 * 1024 * 8 / (bw * 1000))

            print(f"  [{fname}]  {size_mb}MB  true_cca={asset['true_cca']}  "
                  f"est={est_time}s")

            pcap_path = os.path.join(run_dir, f'{fname}.pcap')
            csv_path  = os.path.join(run_dir, f'{fname}.csv')

            # Capture
            ok, pcap_size = capture_one_asset(
                asset, pcap_path, bw, delay, trace_path)

            if not ok:
                print(f"    ERROR: No pcap (size={pcap_size})")
                results.append({
                    'label':      asset['label'],
                    'true_cca':   asset['true_cca'],
                    'pred_cca':   'unknown',
                    'confidence': 0.0,
                    'correct':    False,
                    'size_mb':    size_mb,
                    't':          np.array([0.0]),
                    'bif':        np.array([0.0]),
                })
                continue

            print(f"    pcap: {pcap_size:,}B", end='')

            # Convert to CSV
            rows = pcap_to_csv(pcap_path, csv_path)
            print(f"  rows={rows}", end='')

            if rows < 10:
                print("  → too few packets")
                results.append({
                    'label': asset['label'], 'true_cca': asset['true_cca'],
                    'pred_cca': 'unknown', 'confidence': 0.0,
                    'correct': False, 'size_mb': size_mb,
                    't': np.array([0.0]), 'bif': np.array([0.0]),
                })
                continue

            # Classify
            pred, conf, t, bif = classify_flow(csv_path, gnb, le, rtt_s)
            correct = (pred == asset['true_cca'])
            mark    = '✓' if correct else '✗'
            print(f"  → {pred.upper():<10} conf={conf:.0%}  {mark}")

            results.append({
                'label':      asset['label'],
                'true_cca':   asset['true_cca'],
                'pred_cca':   pred,
                'confidence': conf,
                'correct':    correct,
                'size_mb':    size_mb,
                't':          t,
                'bif':        bif,
            })

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print("\n  Plotting...")
    plot_flows(results, OUT_DIR)
    save_report(results, OUT_DIR)
    print(f"\nAll outputs: {OUT_DIR}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bw',    type=int, default=DEFAULT_BW)
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY)
    args = parser.parse_args()
    run(bw=args.bw, delay=args.delay)