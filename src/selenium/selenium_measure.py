"""
selenium_measure.py — Browser-based CCA measurement using Selenium
Paper reference: Nebby §3.5 and §4.5

FIXES IN THIS VERSION:
  1. --disable-quic added to Chrome flags
       Chrome defaults to HTTP/3 (QUIC) which has its own userspace CCA
       (BBR) that completely ignores the kernel sysctl setting.
       --disable-quic forces Chrome to use TCP, so the kernel CCA applies.

  2. Minimum flow size filter (MIN_FLOW_BYTES = 50 KB)
       Flows smaller than this are HTTP handshakes, favicon fetches,
       preflight requests etc. — they complete too fast to show any
       CCA shape. Filtering them keeps only meaningful flows.

  3. Per-flow BIF_MIN_BYTES lowered to 1000 bytes
       The global preprocess.py threshold (5000B) is tuned for long
       bulk transfers. Individual browser flows are shorter and peak
       at lower BiF — 1000B is the right floor here.

  4. remove_slow_start bypass for short flows
       Flows under 5 seconds don't have a meaningful slow-start /
       congestion-avoidance transition. We skip slow-start removal
       and classify the entire trace directly.

Usage:
    # Terminal 1 — start the asset server:
    python3 selenium_server.py --port 8080

    # Terminal 2 — run measurements:
    python3 selenium_measure.py --cc cubic
    python3 selenium_measure.py --all

Outputs (../evaluation/selenium/):
    selenium_<cc>_flows.png
    selenium_<cc>_summary.txt
    selenium_all_ccas_report.txt
"""

import os, sys, re, glob, time, json, argparse, subprocess
import tempfile, shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../nebby'))

from bif        import compute_bif, smooth_bif
from preprocess import segment_bif
from features   import extract_features
from classify   import detect_bbr, load_model, _majority_vote

# ── config ────────────────────────────────────────────────────────────────────
SERVER_IP   = '10.0.0.1'
SERVER_PORT = 8080
PAGE_URL    = f'http://{SERVER_IP}:{SERVER_PORT}/'

MAHIMAHI_BW    = 2000    # Kbps
MAHIMAHI_DELAY = 50      # ms one-way
RTT_S          = MAHIMAHI_DELAY * 2 / 1000   # 0.1 s

TRACES_DIR = '../traces'
OUT_DIR    = '../evaluation/selenium'
MODEL_DIR  = '../models'

# Flow filtering — ignore flows smaller than this (bytes transferred)
# These are favicon, preflight, keep-alive etc.
MIN_FLOW_BYTES = 50 * 1024   # 50 KB

# BiF floor for per-flow classification — lower than global preprocess.py
# value because browser flows peak at lower BiF than bulk transfers
FLOW_BIF_MIN = 1000   # bytes

# Minimum trace duration to attempt classification (seconds)
MIN_FLOW_DURATION = 1.0

ALL_CCAS = ['cubic', 'reno', 'bbr', 'bic', 'htcp',
            'hybla', 'illinois', 'vegas', 'veno', 'westwood']

ASSET_TYPES = {
    'video':  'video',
    'style':  'static',
    'script': 'static',
    'image':  'static',
    'font':   'static',
    'index':  'page',
}

ASSET_COLORS = {
    'video':          '#e63946',
    'static (large)': '#2196F3',
    'static (small)': '#4CAF50',
    'static':         '#2196F3',
    'page':           '#FF9800',
    'unknown':        '#aaaaaa',
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Mahimahi trace
# ══════════════════════════════════════════════════════════════════════════════

def generate_bw_trace(bw_kbps=MAHIMAHI_BW, duration_s=60,
                      traces_dir=TRACES_DIR):
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
# STEP 2 — Chrome inside Mahimahi
# ══════════════════════════════════════════════════════════════════════════════

def run_chrome_in_mahimahi(cc, pcap_path, har_path,
                            bw=MAHIMAHI_BW, delay=MAHIMAHI_DELAY):

    trace_path = generate_bw_trace(bw)

    # KEY FIX 1: --disable-quic forces Chrome to use TCP
    # Without this, Chrome uses QUIC (HTTP/3) which has its own
    # userspace BBR implementation — completely ignoring sysctl CCA.
    chrome_cmd = (
        'google-chrome '
        '--headless=new '
        '--no-sandbox '
        '--disable-dev-shm-usage '
        '--disable-gpu '
        '--disable-quic '                    # ← FORCE TCP, disable QUIC/HTTP3
        '--disable-background-networking '   # fewer background connections
        '--no-first-run '
        '--no-default-browser-check '
        '--enable-logging=stderr '
        '--log-level=0 '
        f'--virtual-time-budget=20000 '
        f'"{PAGE_URL}" '
        f'2>/tmp/chrome_har.log'
    )

    tcpdump_cmd = (
        f'sudo tcpdump -i ingress -w {pcap_path} -q 2>/dev/null & '
        f'DUMP_PID=$! ; '
        f'sleep 0.5 ; '
        f'{chrome_cmd} ; '
        f'sleep 2 ; '
        f'kill $DUMP_PID 2>/dev/null ; '
        f'wait $DUMP_PID 2>/dev/null'
    )

    mahimahi_cmd = [
        'mm-delay', str(delay),
        'mm-link', trace_path, trace_path,
        '--',
        'bash', '-c', tcpdump_cmd,
    ]

    print(f"  Chrome in Mahimahi (CC={cc}, BW={bw}Kbps, delay={delay}ms, "
          f"QUIC=disabled)...")

    result = subprocess.run(
        mahimahi_cmd,
        capture_output=True, text=True,
        timeout=90,
    )

    if result.returncode != 0:
        print(f"  WARNING: Mahimahi exit code {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:300]}")

    asset_map = _parse_chrome_log('/tmp/chrome_har.log')
    with open(har_path, 'w') as f:
        json.dump(asset_map, f, indent=2)

    return os.path.exists(pcap_path) and os.path.getsize(pcap_path) > 0


def _parse_chrome_log(log_path):
    asset_map = {}
    if not os.path.exists(log_path):
        return asset_map
    try:
        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                for asset_key, asset_type in ASSET_TYPES.items():
                    if asset_key in line.lower():
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
    Split pcap into per-flow CSVs.
    Also returns per-flow byte count so we can filter tiny flows.
    """
    os.makedirs(csv_dir, exist_ok=True)

    # Pass 1: find all flows and their total bytes
    flows_cmd = [
        'tshark', '-r', pcap_path,
        '-Y', 'tcp',
        '-T', 'fields',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'tcp.len',
        '-E', 'separator=,',
    ]
    try:
        result = subprocess.run(flows_cmd, capture_output=True,
                                text=True, timeout=30)
        lines  = result.stdout.strip().split('\n')
    except Exception as e:
        print(f"  tshark error: {e}")
        return []

    # Accumulate per-flow byte counts
    flow_bytes = {}   # (src_ip, src_port) → total_bytes
    seen       = set()
    flow_list  = []

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        src_ip, dst_ip, src_port, dst_port, tcp_len = parts[:5]

        # Server→Client data packets only (these carry payload)
        if src_ip == SERVER_IP and dst_port != str(SERVER_PORT):
            try:
                length = int(tcp_len) if tcp_len else 0
            except ValueError:
                length = 0
            key = (dst_ip, dst_port)   # client IP + client port
            flow_bytes[key] = flow_bytes.get(key, 0) + length

        # Track client→server flows for enumeration
        if dst_ip == SERVER_IP and dst_port == str(SERVER_PORT):
            key = (src_ip, src_port)
            if key not in seen:
                seen.add(key)
                flow_list.append((src_ip, src_port, dst_ip, dst_port))

    total_flows = len(flow_list)

    # KEY FIX 2: filter out tiny flows (< MIN_FLOW_BYTES)
    # These are handshakes, favicon, preflight etc. — no CCA shape visible
    flow_list_filtered = []
    filtered_out       = 0
    for src_ip, src_port, dst_ip, dst_port in flow_list:
        key        = (src_ip, src_port)
        total_b    = flow_bytes.get(key, 0)
        if total_b >= MIN_FLOW_BYTES:
            flow_list_filtered.append(
                (src_ip, src_port, dst_ip, dst_port, total_b))
        else:
            filtered_out += 1

    print(f"  Found {total_flows} flows  →  "
          f"{len(flow_list_filtered)} large enough (≥{MIN_FLOW_BYTES//1024}KB), "
          f"{filtered_out} filtered out (too small)")

    # Pass 2: extract each qualifying flow to CSV
    csv_files = []
    for src_ip, src_port, dst_ip, dst_port, total_b in flow_list_filtered:
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
                               stderr=subprocess.DEVNULL, timeout=30)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 200:
                csv_files.append((csv_path, src_ip, src_port, total_b))
        except Exception as e:
            print(f"  Error extracting flow {src_port}: {e}")

    return csv_files


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify each flow
# ══════════════════════════════════════════════════════════════════════════════

def classify_flow(csv_path, gnb, le,
                  server_ip=SERVER_IP, rtt_s=RTT_S):
    """
    Classify a single browser flow CSV.

    KEY FIX 3: use FLOW_BIF_MIN (1000B) not global BIF_MIN_BYTES (5000B).
    Browser flows are shorter than bulk transfers — peak BiF is lower.

    KEY FIX 4: skip remove_slow_start for short flows.
    Flows under MIN_FLOW_DURATION seconds complete entirely in slow start
    or have only 1-2 oscillations. Cutting 15% of the trace makes things
    worse, not better.
    """
    try:
        t, bif     = compute_bif(csv_path, server_ip)
        t_s, bif_s = smooth_bif(t, bif, rtt_s)
    except Exception as e:
        return 'unknown', 0.0, np.array([0]), np.array([0])

    if len(t_s) < 10:
        return 'unknown', 0.0, t_s, bif_s

    duration = t_s[-1] - t_s[0]

    # For short flows: skip slow-start removal, use full trace
    if duration >= MIN_FLOW_DURATION * 3:
        # Long enough for proper slow-start removal
        from preprocess import remove_slow_start
        t_ss, bif_ss = remove_slow_start(t_s, bif_s)
    else:
        # Short flow — use full trace, just skip the first 10% for noise
        cut          = max(1, len(t_s) // 10)
        t_ss, bif_ss = t_s[cut:], bif_s[cut:]

    # BBR check
    bbr = detect_bbr(t_ss, bif_ss, rtt_s)
    if bbr:
        return 'bbr', 1.0, t_ss, bif_ss

    # Segment with lower BiF floor for browser flows.
    # Per-flow BiF is lower than bulk-transfer BiF because bandwidth
    # is shared across concurrent Chrome connections.
    # At 2000 Kbps / 4 flows = 500 Kbps per flow:
    #   BDP per flow = 500*1000/8 * 0.1s RTT = 6250 bytes
    # So the global BIF_MIN_BYTES=5000 is too high — use 1000 here.
    FLOW_BIF_MIN = 1000   # bytes

    segments = segment_bif(t_ss, bif_ss,
                            drop_fraction=0.35,
                            min_duration_s=0.3,
                            min_points=10,
                            bif_min=FLOW_BIF_MIN)

    # If no segments found, try more aggressive threshold
    if len(segments) == 0:
        segments = segment_bif(t_ss, bif_ss,
                                drop_fraction=0.25,
                                min_duration_s=0.2,
                                min_points=8,
                                bif_min=500)

    feats = extract_features(segments)

    if len(feats) == 0:
        return 'unknown', 0.0, t_ss, bif_ss

    # Handle 6D model
    if gnb.theta_.shape[1] == 6:
        feats = np.hstack([feats, feats])

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)
    return label, conf, t_ss, bif_ss


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Correlate flows to asset types
# ══════════════════════════════════════════════════════════════════════════════

def correlate_assets(flow_results, har_data):
    """
    Map each flow to an asset type using:
      1. HAR data (URL→asset_type) if available
      2. Flow size heuristic:
           Largest flow      → video  (5MB file)
           2nd/3rd largest   → static large  (500KB images)
           Rest              → static small  (CSS, JS, font)
    """
    # Sort by total bytes (largest first)
    flows_with_size = []
    for port, label, conf, t, bif, csv_path, total_b in flow_results:
        flows_with_size.append((total_b, port, label, conf, t, bif))
    flows_with_size.sort(reverse=True)

    annotated = []
    for rank, (size, port, label, conf, t, bif) in \
            enumerate(flows_with_size):

        if rank == 0:
            asset_type = 'video'
        elif rank <= 2:
            asset_type = 'static (large)'
        else:
            asset_type = 'static (small)'

        # Override with HAR if URL contains port
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
# STEP 6 — Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_flows(annotated_flows, cc, out_dir):
    # Only plot classifiable flows
    plottable = [f for f in annotated_flows
                 if len(f['t']) > 5]
    n = len(plottable)
    if n == 0:
        print("  No plottable flows.")
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(7 * cols, 4 * rows))

    for idx, flow in enumerate(plottable):
        ax    = fig.add_subplot(rows, cols, idx + 1)
        t     = flow['t']
        bif   = flow['bif']
        atype = flow['asset_type']
        pred  = flow['pred_cca']
        color = ASSET_COLORS.get(atype, '#aaaaaa')

        bif_roll = (pd.Series(bif)
                    .rolling(10, center=True, min_periods=1)
                    .mean().values)

        ax.fill_between(t, 0, bif / 1024, alpha=0.15, color=color)
        ax.plot(t, bif      / 1024, color=color, lw=0.6, alpha=0.5)
        ax.plot(t, bif_roll / 1024, color=color, lw=2.0)

        ax.set_title(
            f"Port {flow['port']}  |  {atype}\n"
            f"Pred: {pred.upper()}  conf={flow['confidence']:.0%}  "
            f"size={flow['size_kb']:.0f}KB",
            fontsize=8,
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("KB in flight", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=a)
                  for a, c in ASSET_COLORS.items()
                  if a != 'unknown']
    fig.legend(handles=legend_els, fontsize=8,
               loc='lower center', ncol=len(legend_els),
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Selenium — True CCA: {cc.upper()} (TCP, QUIC disabled)\n"
        f"Each subplot = one concurrent TCP connection | "
        f"Colour = asset type",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(out_dir, f'selenium_{cc}_flows.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Report
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(annotated_flows, cc, out_dir):
    classified   = [f for f in annotated_flows if f['pred_cca'] != 'unknown']
    unclassified = [f for f in annotated_flows if f['pred_cca'] == 'unknown']

    lines = [
        f"Nebby Selenium — {cc.upper()}  (QUIC disabled, TCP only)",
        "=" * 60,
        f"True CCA (kernel sysctl): {cc.upper()}",
        "",
        f"{'PORT':<8} {'ASSET TYPE':<20} {'PRED CCA':<14} "
        f"{'CONF':>5}  {'SIZE':>8}  CORRECT?",
        "─" * 65,
    ]

    correct_count = 0
    for flow in annotated_flows:
        correct = (flow['pred_cca'] == cc)
        if flow['pred_cca'] != 'unknown' and correct:
            correct_count += 1
        mark = ('✓' if correct else '✗') if flow['pred_cca'] != 'unknown' \
               else '─'
        lines.append(
            f"{flow['port']:<8} {flow['asset_type']:<20} "
            f"{flow['pred_cca']:<14} {flow['confidence']:>4.0%}  "
            f"{flow['size_kb']:>6.0f}KB  {mark}"
        )

    total_classifiable = len(classified)
    lines += [
        "",
        f"Classifiable flows : {total_classifiable}/{len(annotated_flows)}",
        f"Correct predictions: {correct_count}/{total_classifiable} = "
        f"{correct_count/total_classifiable:.0%}"
        if total_classifiable else "No classifiable flows",
        "",
        "Asset type → CCA mapping (Nebby Table 8 equivalent):",
    ]

    asset_to_cca = {}
    for flow in annotated_flows:
        atype = flow['asset_type']
        asset_to_cca.setdefault(atype, []).append(flow['pred_cca'])

    for atype, ccas in sorted(asset_to_cca.items()):
        classified_ccas = [c for c in ccas if c != 'unknown']
        if classified_ccas:
            from collections import Counter
            most_common = Counter(classified_ccas).most_common(1)[0][0]
            lines.append(f"  {atype:<22}: {most_common}  "
                         f"(from {len(classified_ccas)} classified flows)")

    report = '\n'.join(lines)
    print("\n" + report)

    path = os.path.join(out_dir, f'selenium_{cc}_summary.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {path}")

    return correct_count, total_classifiable


def save_all_report(all_results, out_dir):
    lines = [
        "Nebby Selenium — All CCAs  (QUIC disabled, TCP forced)",
        "Equivalent to paper Table 8 — applied to local controlled setup",
        "=" * 65,
        "",
        f"{'CCA':<12} {'CLASSIFIED':>10}  {'CORRECT':>8}  "
        f"{'ACCURACY':>8}  {'VIDEO CCA':<12} {'STATIC CCA':<12}",
        "─" * 65,
    ]

    for cc, flows, correct, total in all_results:
        acc         = f"{correct/total:.0%}" if total else "N/A"
        video_ccas  = [f['pred_cca'] for f in flows
                       if 'video' in f['asset_type']
                       and f['pred_cca'] != 'unknown']
        static_ccas = [f['pred_cca'] for f in flows
                       if 'static' in f['asset_type']
                       and f['pred_cca'] != 'unknown']
        video_str  = video_ccas[0]  if video_ccas  else '─'
        static_str = static_ccas[0] if static_ccas else '─'
        lines.append(
            f"{cc:<12} {total:>10}  {correct:>8}  "
            f"{acc:>8}  {video_str:<12} {static_str:<12}"
        )

    lines += [
        "",
        "NOVEL CONTRIBUTION vs paper (§4.5):",
        "  Paper observes CCAs on real websites — cannot verify accuracy.",
        "  Here we SET the kernel CCA and VERIFY the prediction,",
        "  giving ground-truth accuracy numbers for browser measurement.",
    ]

    report = '\n'.join(lines)
    print("\n" + "=" * 65)
    print(report)

    path = os.path.join(out_dir, 'selenium_all_ccas_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nSaved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def check_server():
    probe = ['mm-delay', '1', 'bash', '-c',
             f'curl -s --max-time 5 {PAGE_URL} > /dev/null']
    try:
        r = subprocess.run(probe, timeout=15, capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def measure_one_cca(cc, out_dir, gnb, le):
    print(f"\n{'='*55}")
    print(f"  Measuring CC={cc.upper()}  (QUIC disabled)")
    print(f"{'='*55}")

    # Set kernel CCA
    subprocess.run(['sudo', 'sysctl', '-w',
                    f'net.ipv4.tcp_congestion_control={cc}'],
                   capture_output=True)
    actual = subprocess.run(
        ['cat', '/proc/sys/net/ipv4/tcp_congestion_control'],
        capture_output=True, text=True).stdout.strip()

    if actual != cc:
        print(f"  WARNING: Cannot set {cc} (got {actual}) — SKIPPING")
        return None

    run_dir   = tempfile.mkdtemp(prefix=f'nebby_sel_{cc}_')
    pcap_path = os.path.join(run_dir, f'{cc}.pcap')
    har_path  = os.path.join(run_dir, f'{cc}_har.json')
    csv_dir   = os.path.join(run_dir, 'flows')

    try:
        success = run_chrome_in_mahimahi(cc, pcap_path, har_path)
        if not success:
            print(f"  ERROR: No pcap for {cc}")
            return None

        pcap_size = os.path.getsize(pcap_path)
        print(f"  pcap: {pcap_size:,} bytes")

        flow_csvs = split_pcap_by_flow(pcap_path, csv_dir)
        if not flow_csvs:
            print("  No large-enough flows found")
            return None

        flow_results = []
        for csv_path, src_ip, port, total_b in flow_csvs:
            label, conf, t, bif = classify_flow(csv_path, gnb, le)
            size_kb = total_b / 1024
            print(f"    Port {port:<6}  {size_kb:>7.1f}KB  "
                  f"→ {label:<12} conf={conf:.0%}")
            flow_results.append(
                (port, label, conf, t, bif, csv_path, total_b))

        har_data  = {}
        if os.path.exists(har_path):
            with open(har_path) as f:
                har_data = json.load(f)

        annotated = correlate_assets(flow_results, har_data)
        plot_flows(annotated, cc, out_dir)
        correct, total = save_summary(annotated, cc, out_dir)

        return annotated, correct, total

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_all_ccas(out_dir, gnb, le, ccas=ALL_CCAS):
    all_results = []
    for cc in ccas:
        result = measure_one_cca(cc, out_dir, gnb, le)
        if result:
            annotated, correct, total = result
            all_results.append((cc, annotated, correct, total))
    if all_results:
        save_all_report(all_results, out_dir)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc',   default=None)
    parser.add_argument('--all',  action='store_true')
    parser.add_argument('--ccas', nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  Nebby — Selenium Browser Measurement")
    print(f"  Page  : {PAGE_URL}")
    print(f"  BW    : {MAHIMAHI_BW} Kbps  Delay: {MAHIMAHI_DELAY}ms")
    print(f"  QUIC  : DISABLED  (--disable-quic)")
    print(f"  Filter: flows < {MIN_FLOW_BYTES//1024}KB ignored")
    print("=" * 55)

    try:
        gnb, le = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\n  Checking server at {PAGE_URL} ...")
    if not check_server():
        print(f"\nERROR: Server not reachable at {PAGE_URL}")
        print("  Start it first:  python3 selenium_server.py --port 8080")
        sys.exit(1)
    print("  Server OK.\n")

    if args.all:
        run_all_ccas(OUT_DIR, gnb, le)
    elif args.ccas:
        run_all_ccas(OUT_DIR, gnb, le, ccas=args.ccas)
    elif args.cc:
        result = measure_one_cca(args.cc, OUT_DIR, gnb, le)
        if not result:
            sys.exit(1)
    else:
        print("Usage:")
        print("  python3 selenium_measure.py --cc cubic")
        print("  python3 selenium_measure.py --all")
        sys.exit(1)

    print(f"\nAll outputs saved to {OUT_DIR}/")