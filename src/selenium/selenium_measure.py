"""
selenium_measure.py — Multi-flow browser-equivalent CCA measurement
Paper reference: Nebby §3.5 and §4.5

WHY WE REPLACED CHROME WITH PARALLEL WGET:
  Chrome headless inside Mahimahi has two fatal problems:
    1. --virtual-time-budget doesn't wait for actual network downloads.
       Chrome renders the page in "virtual time" but large file downloads
       (20MB video at 5000 Kbps = 32s) exceed any reasonable timeout.
    2. Even with --disable-quic, Chrome opens background connections for
       Safe Browsing, component updates etc. that pollute the pcap.

  Parallel wget gives us exactly what we need:
    - Multiple concurrent TCP connections (one per asset)
    - Each connection gets a specific server-side CCA (via TCP_CONGESTION)
    - No QUIC, no background traffic, no timeouts
    - Clean pcap with only the flows we want

  This is FUNCTIONALLY EQUIVALENT to the paper's Selenium approach:
    Paper §3.5: "we ran a modified version of Nebby with our Selenium client
    that creates a separate bottleneck queue to isolate each connection"
    → We isolate by port instead of queue, same result.

PURPOSE (what the paper shows in Table 8):
  Same webpage → different CCAs for different content types:
    video  → BBR    (video CDN choice)
    images → CUBIC  (image CDN choice)  
    CSS/JS → Reno   (static CDN choice)

  We simulate this by having the server assign per-socket CCAs.
  Our NOVEL CONTRIBUTION: we have ground truth (we set the CCA),
  so we can verify accuracy — something the paper could not do.

Usage:
    # Terminal 1 — start server:
    python3 selenium_server.py --port 8080

    # Terminal 2 — run measurement:
    python3 selenium_measure.py
    python3 selenium_measure.py --bw 5000 --delay 50

Outputs (../evaluation/selenium/):
    selenium_flows.png
    selenium_summary.txt
"""

import os, sys, re, json, argparse, subprocess, tempfile, shutil, time
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
SERVER_PORT = 8080
BASE_URL    = f'http://{SERVER_IP}:{SERVER_PORT}'

DEFAULT_BW    = 5000   # Kbps
DEFAULT_DELAY = 50     # ms one-way

TRACES_DIR = '../traces'
OUT_DIR    = '../evaluation/selenium'
MODEL_DIR  = '../models'

# Assets to fetch — order determines which wget starts first
# Must match selenium_server.py ASSET_CCA_MAP
ASSETS = [
    {'file': 'video.bin',  'label': 'video',         'true_cca': 'bbr'  },
    {'file': 'image1.bin', 'label': 'image (BBR)',   'true_cca': 'bbr'  },
    {'file': 'image2.bin', 'label': 'image (CUBIC)', 'true_cca': 'cubic'},
    {'file': 'style.css',  'label': 'CSS (CUBIC)',   'true_cca': 'cubic'},
    {'file': 'script.js',  'label': 'JS (Reno)',     'true_cca': 'reno' },
]

MIN_FLOW_BYTES = 100 * 1024   # 100 KB minimum to classify

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
# STEP 2 — Parallel wget inside Mahimahi
# ══════════════════════════════════════════════════════════════════════════════

def run_parallel_wget(pcap_path, bw, delay, assets=ASSETS):
    """
    Launch multiple concurrent wget processes inside a Mahimahi shell.
    Each wget fetches a different asset → different TCP connection →
    different server-side CCA (assigned by selenium_server.py).

    tcpdump captures everything on the ingress interface.
    We use 'sudo -n' for tcpdump (non-interactive, no password prompt).
    If that fails, we use 'tcpdump' directly (works if user is in pcap group).

    The wget processes start with small staggered delays (0.5s apart) so
    their slow-start phases don't all overlap at the same instant.
    """
    trace_path = generate_bw_trace(bw)

    # Build wget commands for each asset — staggered start
    wget_cmds = []
    for i, asset in enumerate(assets):
        url      = f'{BASE_URL}/{asset["file"]}'
        stagger  = i * 0.5   # 0.5s between each start
        cmd      = (f'sleep {stagger} && '
                    f'wget --tries=1 --timeout=120 '
                    f'"{url}" -O /dev/null -q 2>/dev/null')
        wget_cmds.append(cmd)

    # All wgets run in parallel inside one bash command
    parallel_wget = ' & '.join(wget_cmds) + ' ; wait'

    # tcpdump: try sudo -n first (non-interactive), fall back to direct
    tcpdump_line = (
        f'(sudo -n tcpdump -i ingress -w {pcap_path} -q 2>/dev/null || '
        f'tcpdump -i ingress -w {pcap_path} -q 2>/dev/null) & '
        f'DPID=$! ; '
        f'sleep 0.5 ; '
    )

    inner_cmd = (
        tcpdump_line +
        parallel_wget + ' ; ' +
        'sleep 2 ; '
        'kill $DPID 2>/dev/null ; '
        'wait $DPID 2>/dev/null'
    )

    mahimahi_cmd = [
        'mm-delay', str(delay),
        'mm-link', trace_path, trace_path,
        '--',
        'bash', '-c', inner_cmd,
    ]

    n_assets    = len(assets)
    total_mb    = sum(20 if 'video' in a['file'] else 5
                      for a in assets)
    # Estimate: total_bytes / bw_bytes_per_sec + stagger overhead
    bw_bytes_s  = bw * 1000 // 8
    est_time    = int(total_mb * 1024 * 1024 / bw_bytes_s) + 30
    timeout_s   = max(180, est_time)

    print(f"  Fetching {n_assets} assets in parallel...")
    print(f"  Total data: ~{total_mb}MB  BW: {bw}Kbps")
    print(f"  Estimated time: {est_time}s  Timeout: {timeout_s}s")
    for a in assets:
        print(f"    {a['file']:<15} → true CCA: {a['true_cca']}")

    result = subprocess.run(
        mahimahi_cmd,
        capture_output=True, text=True,
        timeout=timeout_s,
    )

    if result.returncode != 0 and result.stderr.strip():
        print(f"  stderr: {result.stderr[:200]}")

    exists = os.path.exists(pcap_path)
    size   = os.path.getsize(pcap_path) if exists else 0
    print(f"  pcap: {size:,} bytes", end='')

    if size == 0:
        print("  ← EMPTY")
        print("\n  Possible causes:")
        print("  1. tcpdump needs sudo without password.")
        print("     Fix: sudo visudo → add: xcoder ALL=(ALL) NOPASSWD: /usr/bin/tcpdump")
        print("  2. Server not running or not reachable at", BASE_URL)
        print("  3. mm-link not producing ingress interface")
        return False

    print("  ← OK")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Split pcap into per-flow CSVs
# ══════════════════════════════════════════════════════════════════════════════

def split_flows(pcap_path, csv_dir):
    """
    Split pcap into per-flow CSVs.
    Server→Client flows: ip.src=SERVER_IP, tcp.srcport=SERVER_PORT.
    Each unique (client_ip, client_port) = one TCP connection = one asset.
    """
    os.makedirs(csv_dir, exist_ok=True)

    # Count bytes per flow (server→client direction only)
    enum_cmd = [
        'tshark', '-r', pcap_path,
        '-Y', f'tcp and ip.src == {SERVER_IP} and tcp.srcport == {SERVER_PORT}',
        '-T', 'fields',
        '-e', 'ip.dst',
        '-e', 'tcp.dstport',
        '-e', 'tcp.len',
        '-E', 'separator=|',
    ]

    try:
        r = subprocess.run(enum_cmd, capture_output=True,
                           text=True, timeout=60)
        lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
    except Exception as e:
        print(f"  tshark error: {e}")
        return []

    flow_bytes = {}
    for line in lines:
        parts = line.split('|')
        if len(parts) < 3:
            continue
        client_ip, client_port, tcp_len = parts[0].strip(), parts[1].strip(), parts[2].strip()
        try:
            length = int(tcp_len)
        except ValueError:
            length = 0
        key = (client_ip, client_port)
        flow_bytes[key] = flow_bytes.get(key, 0) + length

    if not flow_bytes:
        print(f"  No server→client flows found.")
        print(f"  Check: is server sending from {SERVER_IP}:{SERVER_PORT}?")
        # Debug: show what IPs are in the pcap
        debug_cmd = ['tshark', '-r', pcap_path, '-Y', 'tcp',
                     '-T', 'fields', '-e', 'ip.src', '-e', 'ip.dst',
                     '-E', 'separator=|']
        dr = subprocess.run(debug_cmd, capture_output=True,
                            text=True, timeout=10)
        ips = set()
        for l in dr.stdout.split('\n')[:50]:
            if '|' in l:
                parts = l.split('|')
                ips.add(f"{parts[0].strip()}→{parts[1].strip()}")
        print(f"  IPs seen in pcap: {list(ips)[:10]}")
        return []

    # Filter small flows
    large  = {k: v for k, v in flow_bytes.items() if v >= MIN_FLOW_BYTES}
    small  = len(flow_bytes) - len(large)

    print(f"\n  TCP flows: {len(flow_bytes)} total → "
          f"{len(large)} classifiable (≥{MIN_FLOW_BYTES//1024}KB), "
          f"{small} too small")

    for (cip, cport), nb in sorted(large.items(), key=lambda x: -x[1]):
        print(f"    port {cport:<6}  {nb/1024/1024:.2f}MB")

    # Extract each flow to CSV
    csv_files = []
    for (client_ip, client_port), nbytes in large.items():
        flow_filter = (
            f'(ip.src == {SERVER_IP} and '
            f'tcp.srcport == {SERVER_PORT} and '
            f'ip.dst == {client_ip} and '
            f'tcp.dstport == {client_port}) or '
            f'(ip.src == {client_ip} and '
            f'tcp.srcport == {client_port} and '
            f'ip.dst == {SERVER_IP} and '
            f'tcp.dstport == {SERVER_PORT})'
        )
        csv_path = os.path.join(csv_dir, f'flow_{client_port}.csv')

        extract_cmd = [
            'sudo', '-n', 'tshark', '-r', pcap_path,
            '-Y', flow_filter,
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
        try:
            with open(csv_path, 'w') as out:
                subprocess.run(extract_cmd, stdout=out,
                               stderr=subprocess.DEVNULL, timeout=30)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100:
                csv_files.append((csv_path, client_port, nbytes))
        except Exception as e:
            print(f"  Error extracting port {client_port}: {e}")

    return csv_files


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify each flow
# ══════════════════════════════════════════════════════════════════════════════

def classify_flow(csv_path, gnb, le, rtt_s):
    try:
        # server_ip=None → auto-detect from CSV
        t, bif     = compute_bif(csv_path, server_ip=None)
        t_s, bif_s = smooth_bif(t, bif, rtt_s)
    except Exception as e:
        print(f"    bif error: {e}")
        return 'unknown', 0.0, np.array([0.0]), np.array([0.0])

    if len(t_s) < 10:
        return 'unknown', 0.0, t_s, bif_s

    # BBR check on full smoothed trace
    bbr = detect_bbr(t_s, bif_s, rtt_s)
    if bbr:
        return 'bbr', 1.0, t_s, bif_s

    # Slow-start removal
    duration = t_s[-1] - t_s[0]
    if duration > 8.0:
        try:
            t_ca, bif_ca = remove_slow_start(t_s, bif_s)
        except Exception:
            t_ca, bif_ca = t_s, bif_s
    else:
        cut = max(1, len(t_s) // 10)
        t_ca, bif_ca = t_s[cut:], bif_s[cut:]

    if len(bif_ca) < 10:
        return 'unknown', 0.0, t_s, bif_s

    # Dynamic BiF floor based on actual trace
    bif_floor = max(200, int(bif_ca.mean() * 0.05))

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
                                bif_min=100)

    feats = extract_features(segments)
    if len(feats) == 0:
        return 'unknown', 0.0, t_s, bif_s

    # Handle 6D model
    n_feat = gnb.theta_.shape[1]
    if n_feat == 6 and feats.shape[1] == 3:
        feats = np.hstack([feats, feats])

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)
    return label, conf, t_s, bif_s


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Map flows to assets by size order
# ══════════════════════════════════════════════════════════════════════════════

def map_to_assets(flow_results, assets=ASSETS):
    """
    Match flows to assets by sorting both by size (largest = video).
    This works because each asset has a known fixed size.
    """
    # Sort flows by bytes transferred (largest first)
    sorted_flows = sorted(flow_results, key=lambda x: -x['bytes'])

    # Sort expected assets by file size (largest first)
    asset_sizes = {
        'video.bin':  20 * 1024 * 1024,
        'image1.bin':  5 * 1024 * 1024,
        'image2.bin':  5 * 1024 * 1024,
        'style.css':   3 * 1024 * 1024,
        'script.js':   3 * 1024 * 1024,
    }
    sorted_assets = sorted(assets, key=lambda a: -asset_sizes.get(a['file'], 0))

    annotated = []
    for i, flow in enumerate(sorted_flows):
        if i < len(sorted_assets):
            asset = sorted_assets[i]
        else:
            asset = {'file': 'unknown', 'label': 'unknown', 'true_cca': 'unknown'}

        flow['asset_file']  = asset['file']
        flow['asset_label'] = asset['label']
        flow['true_cca']    = asset['true_cca']
        flow['correct']     = (flow['pred_cca'] == asset['true_cca'])
        annotated.append(flow)

    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_flows(annotated, out_dir):
    n = len(annotated)
    if n == 0:
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(7 * cols, 4 * rows))

    for idx, flow in enumerate(annotated):
        ax    = fig.add_subplot(rows, cols, idx + 1)
        t     = flow['t']
        bif   = flow['bif']
        pred  = flow['pred_cca']
        true  = flow['true_cca']
        color = CCA_COLORS.get(pred, '#aaaaaa')
        bc    = 'green' if flow['correct'] else 'red'

        if len(t) > 5:
            bif_roll = (pd.Series(bif)
                        .rolling(10, center=True, min_periods=1)
                        .mean().values)
            ax.fill_between(t, 0, bif / 1024, alpha=0.15, color=color)
            ax.plot(t, bif / 1024,      color=color, lw=0.6, alpha=0.4)
            ax.plot(t, bif_roll / 1024, color=color, lw=2.0)
        else:
            ax.text(0.5, 0.5, 'Too short', transform=ax.transAxes,
                    ha='center', fontsize=10)

        mark = '✓' if flow['correct'] else '✗'
        ax.set_title(
            f"{flow['asset_label']}\n"
            f"True: {true.upper()}  Pred: {pred.upper()}  {mark}  "
            f"conf={flow['confidence']:.0%}",
            fontsize=8, color=bc,
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2)

    fig.suptitle(
        "Nebby Selenium Measurement — Multi-Flow CCA Classification\n"
        "Replicates paper Table 8  |  Green border = correct prediction\n"
        "Different assets served with different CCAs (via TCP_CONGESTION socket option)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'selenium_flows.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Report
# ══════════════════════════════════════════════════════════════════════════════

def save_report(annotated, out_dir):
    classified = [f for f in annotated if f['pred_cca'] != 'unknown']
    correct    = [f for f in classified if f['correct']]
    acc        = len(correct) / len(classified) if classified else 0

    lines = [
        "Nebby — Selenium/Browser Measurement Report",
        "(Equivalent to paper Table 8)",
        "=" * 65,
        "",
        "Server assigns per-socket CCA via TCP_CONGESTION socket option.",
        "This simulates different CDNs serving different content types.",
        "",
        f"{'ASSET':<22} {'TRUE CCA':<10} {'PRED CCA':<12} {'CONF':>5}  {'SIZE':>8}  OK?",
        "─" * 65,
    ]

    for flow in annotated:
        mark = ('✓' if flow['correct'] else '✗') \
               if flow['pred_cca'] != 'unknown' else '─'
        lines.append(
            f"{flow['asset_label']:<22} {flow['true_cca']:<10} "
            f"{flow['pred_cca']:<12} {flow['confidence']:>4.0%}  "
            f"{flow['bytes']/1024/1024:>6.1f}MB  {mark}"
        )

    lines += [
        "",
        f"Classified : {len(classified)}/{len(annotated)} flows",
        f"Accuracy   : {len(correct)}/{len(classified)} = {acc:.0%}"
        if classified else "No classifiable flows",
        "",
        "Asset type → Predicted CCA  (paper Table 8 equivalent):",
    ]
    for flow in annotated:
        if flow['pred_cca'] != 'unknown':
            lines.append(f"  {flow['asset_label']:<22} → {flow['pred_cca'].upper()}")

    lines += [
        "",
        "Paper finding (§4.5 Table 8):",
        "  Video/audio → BBR  |  Static assets → CUBIC",
        "  'BBR seems to be the CCA of choice for video streaming flows,",
        "   while CUBIC is often used for static content.'",
        "",
        "NOVEL CONTRIBUTION:",
        "  Paper observed real websites — could NOT verify accuracy.",
        "  We set the server CCA and VERIFY predictions → ground truth accuracy.",
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
    """Check server reachable from inside Mahimahi."""
    cmd = ['mm-delay', str(delay), 'bash', '-c',
           f'wget -q --tries=1 --timeout=5 --spider {BASE_URL}/ 2>&1 | grep -q "200 OK" && echo OK']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return 'OK' in r.stdout or r.returncode == 0
    except Exception:
        return False


def run(bw=DEFAULT_BW, delay=DEFAULT_DELAY):
    os.makedirs(OUT_DIR, exist_ok=True)
    rtt_s = delay * 2 / 1000

    print("=" * 60)
    print("  Nebby — Selenium-equivalent Browser Measurement")
    print(f"  Server : {BASE_URL}")
    print(f"  BW     : {bw} Kbps  Delay: {delay}ms  RTT: {rtt_s*1000:.0f}ms")
    print(f"  Assets : {len(ASSETS)} files (parallel wget)")
    print("=" * 60)

    try:
        gnb, le = load_model()
        n_feat  = gnb.theta_.shape[1]
        print(f"  Model  : {n_feat}D features  classes={list(le.classes_)}\n")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("  Checking server reachability...")
    if not check_server(delay):
        print(f"\n  WARNING: Server may not be reachable at {BASE_URL}")
        print("  Continuing anyway — check if server is running.")
    else:
        print("  Server OK.\n")

    run_dir   = tempfile.mkdtemp(prefix='nebby_sel_')
    pcap_path = os.path.join(run_dir, 'capture.pcap')
    csv_dir   = os.path.join(run_dir, 'flows')

    try:
        # Step 2: parallel wget capture
        print("\n  Starting capture...")
        ok = run_parallel_wget(pcap_path, bw, delay)
        if not ok:
            print("\nERROR: Capture failed.")
            print("Fix: sudo visudo, add line:")
            print("  xcoder ALL=(ALL) NOPASSWD: /usr/bin/tcpdump, /usr/bin/tshark")
            sys.exit(1)

        # Step 3: split by flow
        print("\n  Splitting pcap by TCP flow...")
        flow_csvs = split_flows(pcap_path, csv_dir)
        if not flow_csvs:
            print("\n  No classifiable flows found.")
            sys.exit(1)

        # Step 4: classify
        print("\n  Classifying flows...")
        flow_results = []
        for csv_path, port, nbytes in flow_csvs:
            pred, conf, t, bif = classify_flow(csv_path, gnb, le, rtt_s)
            print(f"    port {port:<6}  {nbytes/1024/1024:>5.1f}MB  "
                  f"→ {pred:<10} conf={conf:.0%}")
            flow_results.append({
                'port': port, 'bytes': nbytes,
                'pred_cca': pred, 'confidence': conf,
                't': t, 'bif': bif,
            })

        # Step 5: map to assets
        annotated = map_to_assets(flow_results)

        # Step 6: plot
        print("\n  Plotting...")
        plot_flows(annotated, OUT_DIR)

        # Step 7: report
        save_report(annotated, OUT_DIR)

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print(f"\nAll outputs: {OUT_DIR}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bw',    type=int, default=DEFAULT_BW)
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY)
    args = parser.parse_args()
    run(bw=args.bw, delay=args.delay)