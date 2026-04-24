"""
classify.py — Classify a single TCP trace
Paper reference: Nebby §3.4 steps 4 & 5 (GNB + BBR rule-based classifier)

Usage:
    python3 classify.py  <path_to_tcp.csv>
    python3 classify.py  ../candidates-measurements/cc-cubic_aqm-droptail_bw-200_buf-20_123_tcp.csv

Classification order (mirrors the paper):
    1. BBR rule-based detector  (checks for ProbeBW / ProbeRTT periodicity)
    2. Loss-based GNB           (polynomial-coefficient clustering)
    3. Unknown                  (if no segments produced)
"""

import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import extract_features

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR = '../models'
SERVER_IP = '10.0.0.1'
RTT_S     = 0.1    # seconds — adjust to match the delay profile used


# ══════════════════════════════════════════════════════════════════════════════
# BBR rule-based detector  (paper §3.4 step 5)
# ══════════════════════════════════════════════════════════════════════════════

def detect_bbr(t, bif, rtt_s=RTT_S):
    """
    Detect BBR by looking for its characteristic periodic probing behaviour.

    BBRv1 — ProbeBW: sending rate increases by 25% every 8 RTTs
             ProbeRTT: backs off every 10 seconds
    BBRv2 — Stable cruise for ~2 s then backs off every 5 seconds

    Returns 'bbr' if detected, else None.
    """
    if len(bif) < 50:
        return None

    # First derivative (rate of change of BiF)
    dbif = np.diff(bif) / np.diff(t)

    p95 = np.percentile(dbif, 95)
    p05 = np.percentile(dbif, 5)

    spike_times = t[1:][dbif > p95]   # ProbeBW upswings
    dip_times   = t[1:][dbif < p05]   # ProbeRTT downswings

    if len(spike_times) < 3:
        return None

    # Expected ProbeBW interval: 8 RTTs
    expected_probe_gap = 8 * rtt_s                 # e.g. 0.8 s for 100 ms RTT
    spike_gaps         = np.diff(spike_times)
    median_spike_gap   = np.median(spike_gaps)

    # Loose tolerance: ± 2× expected (noise in emulated network)
    if median_spike_gap < expected_probe_gap * 3:
        # Check for ProbeRTT dips to confirm it is really BBR
        if len(dip_times) > 1:
            dip_gaps = np.diff(dip_times)
            if np.median(dip_gaps) < 12:        # ~10 s for BBRv1, 5 s for v2
                return 'bbr'

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main classify function
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    gnb_path = os.path.join(MODEL_DIR, 'gnb.pkl')
    le_path  = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    if not os.path.exists(gnb_path) or not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Model not found in {MODEL_DIR}.\n"
            "Run  python3 train.py  first."
        )
    return joblib.load(gnb_path), joblib.load(le_path)


def classify_trace(csv_path, server_ip=SERVER_IP, rtt_s=RTT_S, verbose=True):
    """
    Classify a single trace CSV.

    Returns
    -------
    label      : str  — predicted CCA name or 'unknown'
    confidence : float — fraction of segments agreeing (0.0–1.0)
                         1.0 for BBR (rule-based, no segments)
    """
    gnb, le = load_model()

    # ── preprocessing ─────────────────────────────────────────────────────────
    t, bif         = compute_bif(csv_path, server_ip)
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)

    # ── step 1: BBR rule-based check ──────────────────────────────────────────
    bbr = detect_bbr(t_ss, bif_ss, rtt_s)
    if bbr is not None:
        if verbose:
            print(f"  Result : BBR  (rule-based detector — ProbeBW/ProbeRTT pattern found)")
        return 'bbr', 1.0

    # ── step 2: loss-based GNB ────────────────────────────────────────────────
    segments = segment_bif(t_ss, bif_ss)
    feats    = extract_features(segments)

    if len(feats) == 0:
        if verbose:
            print("  Result : Unknown  (no usable segments extracted)")
        return 'unknown', 0.0

    preds            = gnb.predict(feats)
    unique, counts   = np.unique(preds, return_counts=True)
    best_enc         = unique[np.argmax(counts)]
    confidence       = counts.max() / counts.sum()
    label            = le.inverse_transform([best_enc])[0]

    if verbose:
        print(f"  Result : {label.upper()}  "
              f"(confidence: {confidence:.0%},  {len(feats)} segments)")
        # Show per-segment breakdown
        print("  Segment breakdown:")
        for enc, cnt in zip(unique, counts):
            cls = le.inverse_transform([enc])[0]
            bar = '█' * cnt
            print(f"    {cls:<10} {cnt:>3}  {bar}")

    return label, confidence


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 classify.py <path_to_tcp.csv>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"\nClassifying: {path}")
    print("-" * 60)
    label, conf = classify_trace(path)
    print("-" * 60)
    print(f"Final answer: {label.upper()}  (confidence: {conf:.0%})\n")