"""
curl_measure.py — Simple single-flow CCA measurement using wget/curl
Paper reference: Nebby §3 (methodology), §4.2 (wget-based measurements)

PURPOSE:
  Quick test to verify the classify pipeline works end-to-end before
  running the full Selenium measurement. Tests one URL at a time.

  Also useful for dataset validation — classify a specific existing CSV
  to verify the GNB gives a sensible result.

Usage:
    # Test a specific URL through Mahimahi:
    python3 curl_measure.py --url http://10.0.0.1:8080/video.bin

    # Test with different CCA:
    sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
    python3 curl_measure.py --url http://10.0.0.1:8080/video.bin

    # Classify an existing CSV directly:
    python3 curl_measure.py --csv ../candidates-measurements/cc-cubic_tcp.csv

    # Run all CCAs on the local server:
    python3 curl_measure.py --all

Outputs:
    Prints predicted CCA to stdout.
    Saves pcap and CSV to /tmp/ for inspection.
"""

import os, sys, re, argparse, subprocess, tempfile, time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
for candidate in [_this_dir,
                  os.path.join(_this_dir, '../nebby'),
                  os.path.join(_this_dir, '../../nebby')]:
    if os.path.exists(os.path.join(candidate, 'bif.py')):
        sys.path.insert(0, candidate)
        break

from bif      import compute_bif, smooth_bif
from classify import classify_trace, load_model

# ── config ────────────────────────────────────────────────────────────────────
SERVER_IP   = '10.0.0.1'
SERVER_PORT = 8080
DEFAULT_URL = f'http://{SERVER_IP}:{SERVER_PORT}/video.bin'

BW    = 5000   # Kbps
DELAY = 50     # ms one-way

TRACES_DIR = '../traces'

ALL_CCAS = ['cubic', 'reno', 'bbr', 'bic', 'htcp', 'hybla',
            'illinois', 'vegas', 'veno', 'westwood']


def generate_bw_trace(bw_kbps=BW, traces_dir=TRACES_DIR, duration_s=120):
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
                ct = t * 1000 + i * 1000 // pps
                ct = max(ct, last_t + 1)
                f.write(f"{ct}\n")
                last_t = ct
    print(f"Generated trace: {path}")
    return path


def capture_url(url, delay=DELAY, bw=BW):
    """
    Download a URL through Mahimahi and return the path to a TCP CSV.

    Steps:
      1. mm-delay + mm-link create a controlled bottleneck
      2. tcpdump captures packets on the ingress interface
      3. wget downloads the URL
      4. tshark converts pcap to CSV
    """
    trace_path = generate_bw_trace(bw)
    ts         = int(time.time())
    pcap_path  = f'/tmp/curl_nebby_{ts}.pcap'
    csv_path   = f'/tmp/curl_nebby_{ts}_tcp.csv'

    inner_cmd = (
        f'sudo tcpdump -i ingress -w {pcap_path} -q 2>/dev/null & '
        f'DPID=$! ; '
        f'sleep 0.3 ; '
        f'wget --tries=1 --timeout=60 "{url}" -O /dev/null -q 2>/dev/null ; '
        f'sleep 1 ; '
        f'kill $DPID 2>/dev/null ; '
        f'wait $DPID 2>/dev/null'
    )

    mahimahi_cmd = [
        'mm-delay', str(delay),
        'mm-link', trace_path, trace_path,
        '--',
        'bash', '-c', inner_cmd,
    ]

    print(f"\nDownloading: {url}")
    print(f"BW={bw}Kbps  delay={delay}ms")

    result = subprocess.run(mahimahi_cmd, capture_output=True,
                            text=True, timeout=120)

    if not os.path.exists(pcap_path) or os.path.getsize(pcap_path) == 0:
        print("ERROR: No pcap produced.")
        print("  Is the server running? Is the URL reachable from inside Mahimahi?")
        return None

    pcap_size = os.path.getsize(pcap_path)
    print(f"pcap: {pcap_size:,} bytes")

    # Convert pcap to CSV
    tshark_cmd = [
        'sudo', 'tshark', '-r', pcap_path,
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
        subprocess.run(tshark_cmd, stdout=f,
                       stderr=subprocess.DEVNULL)

    rows = sum(1 for _ in open(csv_path)) - 1
    print(f"CSV: {rows} rows  →  {csv_path}")

    return csv_path


def classify_csv(csv_path, rtt_s=DELAY*2/1000):
    """Classify a single CSV using the single-profile fallback."""
    print(f"\nClassifying: {csv_path}")
    print("-" * 50)
    label, conf = classify_trace(csv_path, server_ip=None, rtt_s=rtt_s)
    print("-" * 50)
    print(f"Result: {label.upper()}  (confidence: {conf:.0%})")
    return label, conf


def run_all(url=DEFAULT_URL, delay=DELAY, bw=BW):
    """Run measurement for all CCAs and print comparison table."""
    print("\n" + "=" * 55)
    print("  Curl Measurement — All CCAs")
    print("=" * 55)

    try:
        load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = []
    for cc in ALL_CCAS:
        print(f"\n{'─'*40}")
        print(f"Setting CCA = {cc}")

        r = subprocess.run(
            ['sudo', 'sysctl', '-w',
             f'net.ipv4.tcp_congestion_control={cc}'],
            capture_output=True)
        actual = subprocess.run(
            ['cat', '/proc/sys/net/ipv4/tcp_congestion_control'],
            capture_output=True, text=True).stdout.strip()

        if actual != cc:
            print(f"  WARNING: {cc} not available (got {actual}) — skipping")
            continue

        csv_path = capture_url(url, delay, bw)
        if not csv_path:
            results.append((cc, 'ERROR', 0.0))
            continue

        label, conf = classify_csv(csv_path, rtt_s=delay*2/1000)
        results.append((cc, label, conf))
        correct = '✓' if label == cc else '✗'
        print(f"  {correct} True: {cc:<12}  Pred: {label:<12}  conf={conf:.0%}")

    # Summary table
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  {'TRUE CCA':<12} {'PREDICTED':<12} {'CONF':>6}  CORRECT?")
    print("  " + "─" * 40)
    correct_count = 0
    for cc, pred, conf in results:
        mark = '✓' if pred == cc else '✗'
        if pred == cc:
            correct_count += 1
        print(f"  {cc:<12} {pred:<12} {conf:>5.0%}  {mark}")
    total = len([r for r in results if r[1] != 'ERROR'])
    if total:
        print(f"\n  Accuracy: {correct_count}/{total} = {correct_count/total:.0%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple single-flow CCA measurement'
    )
    parser.add_argument('--url',   default=DEFAULT_URL,
                        help=f'URL to download (default: {DEFAULT_URL})')
    parser.add_argument('--csv',   default=None,
                        help='Classify an existing CSV directly (skip capture)')
    parser.add_argument('--delay', type=int, default=DELAY,
                        help=f'One-way delay ms (default {DELAY})')
    parser.add_argument('--bw',    type=int, default=BW,
                        help=f'Bandwidth Kbps (default {BW})')
    parser.add_argument('--all',   action='store_true',
                        help='Run all CCAs sequentially')
    args = parser.parse_args()

    if args.csv:
        # Classify existing CSV directly
        classify_csv(args.csv, rtt_s=args.delay * 2 / 1000)

    elif args.all:
        run_all(url=args.url, delay=args.delay, bw=args.bw)

    else:
        # Single URL capture + classify
        try:
            load_model()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        csv_path = capture_url(args.url, args.delay, args.bw)
        if csv_path:
            classify_csv(csv_path, rtt_s=args.delay * 2 / 1000)