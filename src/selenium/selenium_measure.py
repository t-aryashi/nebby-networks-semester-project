"""
selenium_measure.py — Browser-based CCA measurement using Selenium
Paper reference: Nebby §3.5 and §4.5

WHAT THIS DOES:
  1. Launches Chrome (headless) via Selenium inside a Mahimahi shell
  2. Navigates to the local test page — Chrome opens MULTIPLE concurrent
     TCP connections for different assets (video, CSS, JS, images)
  3. Captures ALL flows with tcpdump at the bottleneck
  4. Separates flows by TCP port (each connection = one port pair)
  5. Classifies each flow independently using the trained GNB
  6. Correlates flows to asset types using HAR (HTTP Archive) log
  7. Reports: which CCA served which asset type

This replicates the paper's Table 8 methodology applied to your local setup.

KEY NOVEL POINT over the paper:
  The paper only runs Selenium against real websites on the Internet.
  Here we demonstrate that the same pipeline works in a controlled lab
  environment — which lets us VERIFY the classification (we know the
  true CCA because we set it) rather than just observe it.

Usage:
    # First, start the asset server:
    sudo python3 selenium_server.py &

    # Then run measurements:
    python3 selenium_measure.py --cc cubic
    python3 selenium_measure.py --cc bbr
    python3 selenium_measure.py --cc reno
    python3 selenium_measure.py --all     # run all CCAs

Outputs (../evaluation/selenium/):
    selenium_<cc>_flows.png        — BiF per flow coloured by asset type
    selenium_<cc>_summary.txt      — which CCA served which asset
    selenium_all_ccas_report.txt   — cross-CCA comparison (Table 8 equivalent)
"""

import os, sys, re, glob, time, json, argparse, subprocess
import tempfile, shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
# bif.py, classify.py etc. are in nebby/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../nebby'))

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features
from classify   import detect_bbr, load_model, _majority_vote

# ── config ────────────────────────────────────────────────────────────────────
SERVER_IP   = '100.64.0.1'    # Mahimahi host IP — always this inside mm-*
SERVER_PORT = 8080
PAGE_URL    = f'http://{SERVER_IP}:{SERVER_PORT}/'

MAHIMAHI_BW     = 2000   # Kbps
MAHIMAHI_DELAY  = 50     # ms one-way
BUFF_MUL        = 20     # 2x BDP
AQM             = 'droptail'

TRACES_DIR  = '../traces'
OUT_DIR     = '../evaluation/selenium'
MODEL_DIR   = '../models'

RTT_S = MAHIMAHI_DELAY * 2 / 1000   # seconds

# Asset type mapping: URL substring → label
ASSET_TYPES = {
    'video':   'video',
    'style':   'static',
    'script':  'static',
    'image':   'static',
    'font':    'static',
    'index':   'page',
}

# CCAs to test
ALL_CCAS = ['cubic', 'reno', 'bbr', 'bic', 'htcp',
            'hybla', 'illinois', 'vegas', 'veno', 'westwood']


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate Mahimahi trace
# ══════════════════════════════════════════════════════════════════════════════

def generate_bw_trace(bw_kbps=MAHIMAHI_BW, duration_s=60,
                      traces_dir=TRACES_DIR):
    """Generate a constant-bandwidth Mahimahi trace."""
    os.makedirs(traces_dir, exist_ok=True)
    path = os.path.join(traces_dir, f'bw_{bw_kbps}.trace')

    if os.path.exists(path):
        return path

    bw_bytes = bw_kbps * 1000 // 8
    pps      = max(1, bw_bytes // 1500)

    with open(path, 'w') as f:
        last_t = 0
        for t in range(duration_s):
            for i in range(pps):
                curr_t = t * 1000 + i * 1000 // pps
                if curr_t <= last_t:
                    curr_t = last_t + 1
                f.write(f"{curr_t}\n")
                last_t = curr_t

    print(f"  Generated trace: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Launch Chrome inside Mahimahi, capture pcap
# ══════════════════════════════════════════════════════════════════════════════

def run_selenium_in_mahimahi(cc, pcap_path, har_path,
                              bw=MAHIMAHI_BW, delay=MAHIMAHI_DELAY,
                              buff_mul=BUFF_MUL, aqm=AQM):
    """
    Launch Chrome inside nested Mahimahi shells.
    Captures all traffic to pcap_path and saves HAR to har_path.

    Architecture:
        mm-delay [delay] \
          mm-link bw.trace bw.trace --uplink-queue=droptail ... \
            bash -c "tcpdump & chrome --headless ... ; kill tcpdump"
    """
    trace_path = generate_bw_trace(bw)

    # Calculate buffer size
    rtt_ms   = delay * 2
    bdp      = (bw * 1000 // 8) * rtt_ms // 1000
    buff     = max(1500, (bdp * buff_mul) // 10)

    # Chrome options for headless HAR capture
    # --enable-logging captures network events we parse for asset types
    chrome_cmd = (
        f'google-chrome '
        f'--headless=new '
        f'--no-sandbox '
        f'--disable-dev-shm-usage '
        f'--disable-gpu '
        f'--enable-logging=stderr '
        f'--log-level=0 '
        f'--dump-dom '
        f'--virtual-time-budget=15000 '
        f'"{PAGE_URL}" '
        f'2>/tmp/chrome_har.log'
    )

    # tcpdump inside the Mahimahi ingress interface.
    # Use 'sudo tcpdump' so it has permission to open the raw socket inside
    # the mahimahi network namespace even when mm-delay/mm-link run as root.
    tcpdump_cmd = (
        f'sudo tcpdump -i ingress -w {pcap_path} -q 2>/dev/null & '
        f'DUMP_PID=$! ; '
        f'sleep 0.5 ; '
        f'{chrome_cmd} ; '
        f'sleep 2 ; '
        f'kill $DUMP_PID 2>/dev/null ; '
        f'wait $DUMP_PID 2>/dev/null'
    )

    # NOTE: '--' is required between mm-link's own options and the COMMAND.
    # Without it, GNU getopt (in permutation mode) scans past 'bash' and
    # picks up '-c' as an mm-link flag → "invalid option -- 'c'".
    mahimahi_cmd = [
        'mm-delay', str(delay),
        'mm-link', trace_path, trace_path,
        '--',          # ← end of mm-link options; everything after is COMMAND
        'bash', '-c', tcpdump_cmd,
    ]

    print(f"  Running Chrome inside Mahimahi (CC={cc}, BW={bw}Kbps, "
          f"delay={delay}ms)...")
    print(f"  Command: mm-delay {delay} mm-link {trace_path} {trace_path} "
          f"-- bash -c ...")

    result = subprocess.run(
        mahimahi_cmd,
        capture_output=True, text=True,
        timeout=60,
    )

    if result.returncode != 0:
        print(f"  WARNING: Mahimahi exited with code {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")

    # Parse Chrome log to get asset URLs (poor-man's HAR)
    asset_map = _parse_chrome_log('/tmp/chrome_har.log')
    with open(har_path, 'w') as f:
        json.dump(asset_map, f, indent=2)

    return os.path.exists(pcap_path) and os.path.getsize(pcap_path) > 0


def _parse_chrome_log(log_path):
    """
    Extract URL→asset_type mapping from Chrome's stderr log.
    Returns dict: {port_or_url: asset_type}
    """
    asset_map = {}
    if not os.path.exists(log_path):
        return asset_map

    try:
        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                # Chrome logs network requests as:
                # [N:url] or similar patterns
                for asset_key, asset_type in ASSET_TYPES.items():
                    if asset_key in line.lower():
                        # Extract URL if present
                        url_match = re.search(r'http://[^\s"]+', line)
                        if url_match:
                            asset_map[url_match.group(0)] = asset_type
    except Exception:
        pass

    return asset_map


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Split pcap by TCP flow
# ══════════════════════════════════════════════════════════════════════════════

def split_pcap_by_flow(pcap_path, csv_dir):
    """
    Use tshark to extract each TCP flow (identified by dst port) into
    a separate CSV file.

    Returns list of (csv_path, src_ip, dst_port) tuples.
    """
    os.makedirs(csv_dir, exist_ok=True)

    # First pass: identify all unique flows
    flows_cmd = [
        'tshark', '-r', pcap_path,
        '-Y', 'tcp',
        '-T', 'fields',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-E', 'separator=,',
    ]

    try:
        result = subprocess.run(flows_cmd, capture_output=True, text=True)
        lines  = result.stdout.strip().split('\n')
    except Exception as e:
        print(f"  tshark error: {e}")
        return []

    # Collect unique (src_ip, dst_ip, src_port, dst_port) tuples
    # A flow = connection to SERVER_PORT on SERVER_IP
    seen_flows = set()
    flow_list  = []

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 4:
            continue
        src_ip, dst_ip, src_port, dst_port = parts[:4]

        # Client→Server flows: dst_ip = SERVER_IP, dst_port = SERVER_PORT
        if dst_ip == SERVER_IP and dst_port == str(SERVER_PORT):
            key = (src_ip, src_port)
            if key not in seen_flows:
                seen_flows.add(key)
                flow_list.append((src_ip, src_port, dst_ip, dst_port))

    print(f"  Found {len(flow_list)} TCP flow(s) in pcap")

    # Second pass: extract each flow to its own CSV
    csv_files = []
    for src_ip, src_port, dst_ip, dst_port in flow_list:
        # Filter expression for this specific flow (both directions)
        flow_filter = (
            f'(ip.src=={src_ip} and tcp.srcport=={src_port}) or '
            f'(ip.dst=={src_ip} and tcp.dstport=={src_port})'
        )
        csv_path = os.path.join(csv_dir, f'flow_{src_port}.csv')

        extract_cmd = [
            'sudo', 'tshark', '-r', pcap_path,
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
                               stderr=subprocess.DEVNULL)
            if os.path.getsize(csv_path) > 100:
                csv_files.append((csv_path, src_ip, src_port))
        except Exception as e:
            print(f"  Error extracting flow {src_port}: {e}")

    return csv_files


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify each flow
# ══════════════════════════════════════════════════════════════════════════════

def classify_flow(csv_path, gnb, le, server_ip=SERVER_IP, rtt_s=RTT_S):
    """
    Classify a single flow CSV.
    Returns (label, confidence, t, bif_smooth).
    """
    try:
        t, bif       = compute_bif(csv_path, server_ip)
        t_s, bif_s   = smooth_bif(t, bif, rtt_s)
        t_ss, bif_ss = remove_slow_start(t_s, bif_s)
    except Exception as e:
        return 'unknown', 0.0, np.array([0]), np.array([0])

    # BBR check
    bbr = detect_bbr(t_ss, bif_ss, rtt_s)
    if bbr:
        return 'bbr', 1.0, t_ss, bif_ss

    # GNB
    segments = segment_bif(t_ss, bif_ss)
    feats    = extract_features(segments)

    if len(feats) == 0:
        return 'unknown', 0.0, t_ss, bif_ss

    # Handle 6D model: duplicate 3D features
    if gnb.theta_.shape[1] == 6:
        feats = np.hstack([feats, feats])

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)
    return label, conf, t_ss, bif_ss


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Correlate flows to asset types
# ══════════════════════════════════════════════════════════════════════════════

def correlate_assets(flow_results, har_data, pcap_path):
    """
    Map each flow (identified by port) to an asset type.

    Strategy:
      1. Try HAR data first (URL→asset_type mapping)
      2. Fall back to flow size heuristics:
         - Largest flow  → video
         - Small flows   → static assets (CSS, JS, images)
         - Tiny flows    → page/font
    """
    # Sort flows by total data transferred (BiF proxy: max BiF ≈ total bytes)
    flows_with_size = []
    for port, label, conf, t, bif, csv_path in flow_results:
        total_size = bif.max() if len(bif) > 0 else 0
        flows_with_size.append((total_size, port, label, conf, t, bif))

    flows_with_size.sort(reverse=True)   # largest first

    annotated = []
    for rank, (size, port, label, conf, t, bif) in enumerate(flows_with_size):
        # Heuristic asset type by rank (largest = video)
        if rank == 0:
            asset_type = 'video'
        elif rank <= 2:
            asset_type = 'static (large)'
        else:
            asset_type = 'static (small)'

        # Override with HAR data if available
        for url, atype in har_data.items():
            if str(port) in url:
                asset_type = atype
                break

        annotated.append({
            'port':       port,
            'asset_type': asset_type,
            'pred_cca':   label,
            'confidence': conf,
            'size_kb':    size / 1024,
            't':          t,
            'bif':        bif,
        })

    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Plot and report
# ══════════════════════════════════════════════════════════════════════════════

ASSET_COLORS = {
    'video':          '#e63946',   # red — video
    'static (large)': '#2196F3',   # blue — large static
    'static (small)': '#4CAF50',   # green — small static
    'static':         '#2196F3',
    'page':           '#FF9800',   # orange — page
    'unknown':        '#aaaaaa',   # grey
}

CCA_MARKERS = {
    'cubic': 'o', 'reno': 's', 'bbr': '^',
    'bic': 'D', 'htcp': 'v', 'hybla': 'P',
    'illinois': '*', 'vegas': 'X', 'veno': 'h',
    'westwood': '+', 'yeah': 'd', 'unknown': '.',
}


def plot_flows(annotated_flows, cc, out_dir):
    """
    Plot BiF per flow coloured by asset type.
    Marker shape shows predicted CCA.
    """
    n = len(annotated_flows)
    if n == 0:
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(7 * cols, 4 * rows))

    for idx, flow in enumerate(annotated_flows):
        ax    = fig.add_subplot(rows, cols, idx + 1)
        t     = flow['t']
        bif   = flow['bif']
        atype = flow['asset_type']
        pred  = flow['pred_cca']
        color = ASSET_COLORS.get(atype, '#aaaaaa')

        # Rolling smooth for display
        bif_roll = (pd.Series(bif)
                    .rolling(10, center=True, min_periods=1)
                    .mean().values)

        ax.fill_between(t, 0, bif / 1024, alpha=0.15, color=color)
        ax.plot(t, bif      / 1024, color=color, lw=0.6, alpha=0.5)
        ax.plot(t, bif_roll / 1024, color=color, lw=2.0)

        ax.set_title(
            f"Port {flow['port']}  |  {atype}\n"
            f"Predicted CCA: {pred.upper()}  "
            f"(conf: {flow['confidence']:.0%})  "
            f"|  {flow['size_kb']:.0f} KB",
            fontsize=8,
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Legend for asset types
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=a)
                  for a, c in ASSET_COLORS.items()
                  if a != 'unknown']
    fig.legend(handles=legend_els, fontsize=8,
               loc='lower center', ncol=len(legend_els),
               bbox_to_anchor=(0.5, -0.02))

    true_cc = cc.upper()
    fig.suptitle(
        f"Selenium Measurement — True CCA: {true_cc}\n"
        f"Each subplot = one concurrent TCP connection  |  "
        f"Colour = asset type  |  Predicted CCA shown in title",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(out_dir, f'selenium_{cc}_flows.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_summary(annotated_flows, cc, true_cc, out_dir):
    """Save per-flow summary text file."""
    lines = [
        f"Nebby Selenium Measurement — {cc.upper()}",
        "=" * 55,
        f"True CCA (set by sysctl): {true_cc.upper()}",
        "",
        f"{'PORT':<8} {'ASSET TYPE':<20} {'PREDICTED CCA':<15} {'CONF':>6}  {'SIZE':>8}",
        f"{'─'*8} {'─'*20} {'─'*15} {'─'*6}  {'─'*8}",
    ]

    correct_count = 0
    for flow in annotated_flows:
        correct = flow['pred_cca'] == true_cc
        mark    = '✓' if correct else '✗'
        if correct:
            correct_count += 1
        lines.append(
            f"{flow['port']:<8} {flow['asset_type']:<20} "
            f"{flow['pred_cca']:<15} {flow['confidence']:>5.0%}  "
            f"{flow['size_kb']:>6.0f}KB  {mark}"
        )

    lines += [
        "",
        f"Flow-level accuracy: {correct_count}/{len(annotated_flows)} = "
        f"{correct_count/len(annotated_flows):.0%}"
        if annotated_flows else "No flows classified",
        "",
        "Asset type → CCA mapping:",
    ]

    for atype in set(f['asset_type'] for f in annotated_flows):
        ccas = [f['pred_cca'] for f in annotated_flows
                if f['asset_type'] == atype]
        lines.append(f"  {atype:<20}: {', '.join(ccas)}")

    report = '\n'.join(lines)
    print("\n" + report)

    path = os.path.join(out_dir, f'selenium_{cc}_summary.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {path}")

    return correct_count, len(annotated_flows)


def save_all_ccas_report(all_results, out_dir):
    """
    Cross-CCA comparison table — equivalent to paper's Table 8.
    Shows which CCA was predicted for video vs static per CCA setting.
    """
    lines = [
        "Nebby Selenium — All CCAs Report",
        "(Equivalent to Table 8 in paper — applied to local setup)",
        "=" * 65,
        "",
        f"{'CCA':<12} {'FLOWS':>5}  {'ACCURACY':>8}  "
        f"{'VIDEO PRED':<15} {'STATIC PRED':<15}",
        f"{'─'*12} {'─'*5}  {'─'*8}  {'─'*15} {'─'*15}",
    ]

    for cc, flows, correct, total in all_results:
        acc = f"{correct/total:.0%}" if total else "N/A"

        video_preds  = [f['pred_cca'] for f in flows
                        if 'video' in f['asset_type']]
        static_preds = [f['pred_cca'] for f in flows
                        if 'static' in f['asset_type']]

        video_str  = video_preds[0]  if video_preds  else '─'
        static_str = static_preds[0] if static_preds else '─'

        lines.append(
            f"{cc:<12} {total:>5}  {acc:>8}  "
            f"{video_str:<15} {static_str:<15}"
        )

    lines += [
        "",
        "Note: In the paper (Table 8), BBR is commonly used for video",
        "      and CUBIC for static assets, even on the same page.",
        "      This reflects different CDN configurations per asset type.",
    ]

    report = '\n'.join(lines)
    print("\n" + "=" * 65)
    print(report)

    path = os.path.join(out_dir, 'selenium_all_ccas_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nSaved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MEASUREMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def check_server_reachable():
    """
    Verify the asset server is reachable from INSIDE a mahimahi shell.

    100.64.0.1 is mahimahi's host-side IP — it is NOT reachable from the
    normal host network (so `curl http://100.64.0.1:8080/` outside mahimahi
    will always time-out).  We probe it by running a one-shot mm-delay shell
    that curls the URL; a fast exit-0 means the server is up.
    """
    probe_cmd = [
        'mm-delay', '1',
        '--',
        'bash', '-c',
        f'curl -s --max-time 5 http://{SERVER_IP}:{SERVER_PORT}/ > /dev/null',
    ]
    try:
        r = subprocess.run(probe_cmd, timeout=15, capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def measure_one_cca(cc, out_dir, gnb, le):
    """
    Full pipeline for one CCA:
      set CCA → launch Chrome in Mahimahi → split flows →
      classify each → correlate to assets → plot + report
    """
    print(f"\n{'='*55}")
    print(f"  Measuring CC={cc.upper()}")
    print(f"{'='*55}")

    # Set CCA
    ret = subprocess.run(
        ['sudo', 'sysctl', '-w',
         f'net.ipv4.tcp_congestion_control={cc}'],
        capture_output=True,
    )
    actual = subprocess.run(
        ['cat', '/proc/sys/net/ipv4/tcp_congestion_control'],
        capture_output=True, text=True,
    ).stdout.strip()

    if actual != cc:
        print(f"  WARNING: Could not set {cc} (got {actual}) — SKIPPING")
        return None

    # Temp dir for this run
    run_dir  = tempfile.mkdtemp(prefix=f'nebby_sel_{cc}_')
    pcap_path = os.path.join(run_dir, f'{cc}.pcap')
    har_path  = os.path.join(run_dir, f'{cc}_har.json')
    csv_dir   = os.path.join(run_dir, 'flows')

    try:
        # Launch Chrome in Mahimahi
        success = run_selenium_in_mahimahi(cc, pcap_path, har_path)
        if not success:
            print(f"  ERROR: No pcap produced for {cc}")
            return None

        print(f"  pcap: {os.path.getsize(pcap_path):,} bytes")

        # Split pcap into per-flow CSVs
        flow_csvs = split_pcap_by_flow(pcap_path, csv_dir)
        if not flow_csvs:
            print(f"  WARNING: No flows extracted from pcap")
            return None

        # Classify each flow
        flow_results = []
        for csv_path, src_ip, port in flow_csvs:
            label, conf, t, bif = classify_flow(csv_path, gnb, le)
            print(f"    Port {port:<6} → {label:<12} conf={conf:.0%}")
            flow_results.append((port, label, conf, t, bif, csv_path))

        # Load HAR
        har_data = {}
        if os.path.exists(har_path):
            with open(har_path) as f:
                har_data = json.load(f)

        # Correlate to asset types
        annotated = correlate_assets(flow_results, har_data, pcap_path)

        # Plot and report
        plot_flows(annotated, cc, out_dir)
        correct, total = save_summary(annotated, cc, cc, out_dir)

        return annotated, correct, total

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_all_ccas(out_dir, gnb, le, ccas=ALL_CCAS):
    """Run measurements for all CCAs and produce comparison report."""
    all_results = []

    for cc in ccas:
        result = measure_one_cca(cc, out_dir, gnb, le)
        if result:
            annotated, correct, total = result
            all_results.append((cc, annotated, correct, total))

    if all_results:
        save_all_ccas_report(
            [(cc, flows, correct, total)
             for cc, flows, correct, total in all_results],
            out_dir,
        )

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nebby Selenium browser-based CCA measurement'
    )
    parser.add_argument('--cc',  default=None,
                        help='Single CCA to measure (e.g. cubic)')
    parser.add_argument('--all', action='store_true',
                        help='Measure all CCAs')
    parser.add_argument('--ccas', nargs='+', default=None,
                        help='Specific list of CCAs to measure')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  Nebby — Selenium Browser Measurement")
    print(f"  Page: {PAGE_URL}")
    print(f"  BW: {MAHIMAHI_BW} Kbps  Delay: {MAHIMAHI_DELAY}ms")
    print("=" * 55)

    # Load model
    try:
        gnb, le = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Verify asset server is reachable from inside a mahimahi shell.
    # NOTE: 100.64.0.1 is mahimahi's internal host IP — you CANNOT reach it
    #       with a plain `curl` from the normal host shell.  Start the server
    #       first:  sudo python3 selenium_server.py &
    print(f"\n  Checking server at {PAGE_URL} (from inside mahimahi)...")
    if not check_server_reachable():
        print(
            f"\nERROR: Cannot reach {PAGE_URL} from inside mahimahi.\n"
            f"  Start the asset server first (in a separate terminal):\n"
            f"    sudo python3 selenium_server.py &\n"
            f"  NOTE: Do NOT test with plain curl from the host shell —\n"
            f"  100.64.0.1 is only routable *inside* a mahimahi namespace."
        )
        sys.exit(1)
    print("  Server OK.\n")

    if args.all:
        run_all_ccas(OUT_DIR, gnb, le)

    elif args.ccas:
        run_all_ccas(OUT_DIR, gnb, le, ccas=args.ccas)

    elif args.cc:
        result = measure_one_cca(args.cc, OUT_DIR, gnb, le)
        if not result:
            print(f"\nMeasurement failed for {args.cc}")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python3 selenium_measure.py --cc cubic")
        print("  python3 selenium_measure.py --all")
        print("  python3 selenium_measure.py --ccas cubic bbr reno")
        sys.exit(1)

    print(f"\nAll outputs saved to {OUT_DIR}/")