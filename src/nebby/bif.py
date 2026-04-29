"""
bif.py — Bytes-in-Flight computation and smoothing
Paper reference: Nebby §3.1 (Estimating BiF) and §3.4 step 1 (Smoothening)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def detect_server_ip(csv_path):
    """
    Auto-detect the server IP from a CSV by finding which IP sent
    the most payload bytes (highest sum of tcp.len).
    Used for real internet traffic where server_ip is not known in advance.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['tcp.len'] = pd.to_numeric(df['tcp.len'], errors='coerce').fillna(0)
    df = df.dropna(subset=['ip.src'])

    data_pkts = df[df['tcp.len'] > 0]

    if data_pkts.empty:
        raise ValueError("No data packets found")

    counts = data_pkts['ip.src'].value_counts()
    server_ip = counts.idxmax()
    return server_ip


def compute_bif(csv_path, server_ip=None):
    """
    Compute raw Bytes-in-Flight from a tshark CSV.

    BiF(t) = cummax(tcp.seq + tcp.len)  -  cummax(tcp.ack)
             |__ highest byte sent __|     |__ highest byte ACKed __|

    Parameters
    ----------
    csv_path  : path to *_tcp.csv produced by pcap2csv.sh
    server_ip : IP of the TCP sender (your HTTP server)

    Returns
    -------
    t   : numpy array of timestamps (seconds)
    bif : numpy array of BiF values (bytes)
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    for col in ['tcp.len', 'tcp.seq', 'tcp.ack', 'frame.time_relative']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['frame.time_relative', 'ip.src'])
    df = df.sort_values('frame.time_relative').reset_index(drop=True)

    # Auto-detect server IP if not provided
    if server_ip is None:
        server_ip = detect_server_ip(csv_path)
        print(f"  Auto-detected server IP: {server_ip}")

    srv = df[df['ip.src'] == server_ip].copy()   # data packets (server → client)
    cli = df[df['ip.src'] != server_ip].copy()   # ACK  packets (client → server)

    if srv.empty or cli.empty:
        raise ValueError(
            f"Missing one direction in {csv_path}.\n"
            f"  server rows : {len(srv)}\n"
            f"  client rows : {len(cli)}\n"
            f"  Check server_ip='{server_ip}'"
        )

    # Running max of (seq + len) — highest byte the server has ever sent
    srv['seq_end']     = srv['tcp.seq'] + srv['tcp.len']
    srv['max_seq_end'] = srv['seq_end'].cummax()

    # Running max of ACK — highest byte the client has confirmed received
    cli['max_ack'] = cli['tcp.ack'].cummax()

    t   = srv['frame.time_relative'].values
    msa = np.interp(
        t,
        cli['frame.time_relative'].values,
        cli['max_ack'].values,
        left=cli['max_ack'].values[0],
        right=cli['max_ack'].values[-1],
    )

    bif = np.maximum(srv['max_seq_end'].values - msa, 0)
    return t, bif


def smooth_bif(t, bif, rtt_s=0.1):
    """
    Low-pass filter: remove all variation faster than 1/RTT.
    (Paper §3.4 step 1 — FFT-based smoothening)

    Any noise faster than one RTT is caused by the network
    (ACK compression, reordering) not by the CCA itself.

    Parameters
    ----------
    t     : timestamps from compute_bif
    bif   : raw BiF values from compute_bif
    rtt_s : estimated RTT in seconds
              50 ms delay trace  → rtt_s = 0.10  (2 × 50 ms)
              100 ms delay trace → rtt_s = 0.20  (2 × 100 ms)

    Returns
    -------
    t_uni      : uniformly-spaced timestamps
    bif_smooth : smoothed BiF values (bytes)
    """
    # Resample to uniform time grid (required for digital filter)
    t_uni   = np.linspace(t[0], t[-1], len(t))
    bif_uni = np.interp(t_uni, t, bif)

    dt     = (t_uni[-1] - t_uni[0]) / max(len(t_uni) - 1, 1)
    fs     = 1.0 / dt if dt > 0 else 1000.0   # sampling frequency (Hz)
    cutoff = 1.0 / rtt_s                        # e.g. 10 Hz for 100 ms RTT
    wn     = min(cutoff / (fs / 2.0), 0.99)    # normalised cut-off [0,1)

    b, a       = butter(4, wn, btype='low')
    bif_smooth = filtfilt(b, a, bif_uni)

    return t_uni, np.maximum(bif_smooth, 0)