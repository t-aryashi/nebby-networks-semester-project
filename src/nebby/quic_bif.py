"""
quic_bif.py — Bytes-in-Flight estimation for QUIC flows
Paper reference: Nebby §3.1 "Handling QUIC" + §3.4 step 1

WHY THIS FILE EXISTS
━━━━━━━━━━━━━━━━━━━
bif.py uses TCP sequence and ACK numbers to compute BiF precisely:
    BiF(t) = max(tcp.seq + tcp.len)  −  max(tcp.ack)

QUIC encrypts everything. tshark cannot parse QUIC sequence numbers
or ACK numbers — it only sees raw UDP datagrams with known payload size.

This file implements the two-assumption workaround from the paper:

  Assumption 1 — Direction labels:
    All server→client UDP packets carry QUIC DATA.
    All client→server UDP packets carry QUIC ACKs.

  Assumption 2 — Constant ACK granularity:
    Every ACK packet acknowledges a fixed number of bytes:
        bytes_per_ack = Σ(server payload bytes) / count(client packets)
    This holds because most QUIC stacks ACK every N data packets
    throughout a connection (N is typically 2 for most stacks).

  BiF(t) = Σ_server(udp.length up to t)
           − count_client(packets up to t) × bytes_per_ack

  Clamped to ≥ 0 to absorb small estimation error.

The paper validated this against ground-truth BiF exported from quiche
sender sockets: accuracy > 97% across 20 trials in 2 AWS regions.

TSHARK CSV FORMAT
━━━━━━━━━━━━━━━━
This file expects a CSV produced by a tshark command like:

    tshark -r capture.pcap \\
           -Y "udp" \\
           -T fields \\
           -e frame.time_relative \\
           -e ip.src \\
           -e ip.dst \\
           -e udp.length \\
           -E header=y -E separator=, -E quote=d \\
           > quic_capture.csv

Note: tshark's udp.length includes the 8-byte UDP header, so actual
payload = udp.length − 8. We subtract it consistently for both
server and client packets, so it cancels out in the BiF calculation
(bytes_per_ack is estimated from the same raw udp.length values).

OUTPUT
━━━━━━
Returns (t, bif) — identical types to compute_bif() in bif.py.
You can then pass these directly into smooth_bif(), remove_slow_start(),
segment_bif(), extract_features(), and classify_trace_pair() unchanged.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Server IP detection (identical logic to bif.py, adapted for UDP)
# ─────────────────────────────────────────────────────────────────────────────

def detect_server_ip_quic(csv_path):
    """
    Auto-detect the server IP from a QUIC UDP CSV.

    The server sends the most bytes (it's streaming data to the client),
    so we find the IP with the highest total udp.length.

    Parameters
    ----------
    csv_path : path to *_quic.csv produced by pcap2csv_quic.sh

    Returns
    -------
    server_ip : str
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df['udp.length'] = pd.to_numeric(df['udp.length'], errors='coerce').fillna(0)
    df = df.dropna(subset=['ip.src'])

    # Server sends large data packets; client sends small ACK packets.
    # Summing udp.length per source IP reliably identifies the server.
    totals    = df.groupby('ip.src')['udp.length'].sum()
    server_ip = totals.idxmax()
    return server_ip


# ─────────────────────────────────────────────────────────────────────────────
# Core BiF computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_bif_quic(csv_path, server_ip=None):
    """
    Compute raw Bytes-in-Flight from a QUIC tshark UDP CSV.

    Parameters
    ----------
    csv_path  : path to *_quic.csv  (columns: frame.time_relative,
                ip.src, ip.dst, udp.length)
    server_ip : IP of the QUIC server (sender of data).
                If None, auto-detected via detect_server_ip_quic().

    Returns
    -------
    t   : numpy array of timestamps (seconds) — one entry per server packet
    bif : numpy array of BiF values (bytes)   — same length as t

    These have the same types and semantics as the output of compute_bif()
    in bif.py, so you can pass them directly into smooth_bif().
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df['udp.length']          = pd.to_numeric(df['udp.length'],          errors='coerce').fillna(0)
    df['frame.time_relative'] = pd.to_numeric(df['frame.time_relative'], errors='coerce')

    df = df.dropna(subset=['frame.time_relative', 'ip.src'])
    df = df.sort_values('frame.time_relative').reset_index(drop=True)

    # ── Auto-detect server IP if not provided ─────────────────────────────────
    if server_ip is None:
        server_ip = detect_server_ip_quic(csv_path)
        print(f"  Auto-detected QUIC server IP: {server_ip}")

    # ── Split into server→client and client→server packets ───────────────────
    srv = df[df['ip.src'] == server_ip].copy()    # DATA  packets
    cli = df[df['ip.src'] != server_ip].copy()    # ACK   packets

    if srv.empty or cli.empty:
        raise ValueError(
            f"Missing one direction in {csv_path}.\n"
            f"  server rows : {len(srv)}\n"
            f"  client rows : {len(cli)}\n"
            f"  Check server_ip='{server_ip}'"
        )

    # ── Assumption 2: estimate bytes_per_ack ─────────────────────────────────
    # udp.length includes the 8-byte UDP header; actual payload = len - 8.
    # We apply the same correction to both sides, so it cancels in the ratio,
    # but being explicit is cleaner and handles edge cases.
    HEADER = 8   # UDP header bytes
    total_server_payload = max((srv['udp.length'] - HEADER).clip(lower=0).sum(), 1)
    total_ack_packets    = len(cli)

    bytes_per_ack = total_server_payload / total_ack_packets

    print(f"  QUIC bytes_per_ack estimate : {bytes_per_ack:.1f} B")
    print(f"  Server packets : {len(srv)}   "
          f"Client (ACK) packets : {total_ack_packets}")

    # ── Rolling BiF ───────────────────────────────────────────────────────────
    # Merge all events into one sorted timeline and compute running sums.
    srv_events = pd.DataFrame({
        'time': srv['frame.time_relative'].values,
        'direction': 'srv',
        'payload':   (srv['udp.length'] - HEADER).clip(lower=0).values,
    })
    cli_events = pd.DataFrame({
        'time': cli['frame.time_relative'].values,
        'direction': 'cli',
        'payload':   np.ones(len(cli)),    # 1 ACK packet each
    })

    events = pd.concat([srv_events, cli_events], ignore_index=True)
    events = events.sort_values('time').reset_index(drop=True)

    cum_server_bytes = 0.0
    cum_ack_pkts     = 0.0

    t_list   = []
    bif_list = []

    for _, row in events.iterrows():
        if row['direction'] == 'srv':
            cum_server_bytes += row['payload']
        else:
            cum_ack_pkts     += 1.0

        bif_val = cum_server_bytes - cum_ack_pkts * bytes_per_ack
        bif_val = max(0.0, bif_val)   # clamp: Assumption 2 is approximate

        # Only emit a sample on server-packet events (same as bif.py
        # which only timestamps server-sent packets)
        if row['direction'] == 'srv':
            t_list.append(row['time'])
            bif_list.append(bif_val)

    t   = np.array(t_list,   dtype=float)
    bif = np.array(bif_list, dtype=float)

    return t, bif


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper (optional — requires ground-truth quiche stats JSON)
# ─────────────────────────────────────────────────────────────────────────────

def validate_bif_accuracy(estimated_bif, groundtruth_json_path):
    """
    Compare estimated BiF against the ground-truth BiF exported by
    quiche's --dump-json flag.

    The paper reports > 97% accuracy using this metric.

    Parameters
    ----------
    estimated_bif          : numpy array from compute_bif_quic()
    groundtruth_json_path  : path to JSON with {"stats": [{"bytes_in_flight": ...}, ...]}

    Returns
    -------
    accuracy_pct : float — e.g. 97.3
    """
    import json

    with open(groundtruth_json_path) as fh:
        stats = json.load(fh)

    gt_bif = np.array([e['bytes_in_flight'] for e in stats.get('stats', [])])
    if len(gt_bif) == 0:
        raise ValueError("Ground-truth JSON has no 'stats' entries")

    n = min(len(estimated_bif), len(gt_bif))
    if n == 0:
        return 0.0

    e = estimated_bif[:n]
    g = gt_bif[:n]

    # Mean absolute percentage error (ignore points where GT ~ 0)
    mask = g > 100
    if not np.any(mask):
        return 0.0

    mape     = np.mean(np.abs(e[mask] - g[mask]) / g[mask]) * 100.0
    accuracy = max(0.0, 100.0 - mape)
    return accuracy