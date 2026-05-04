"""
classify_quic.py — Classify a QUIC trace (or dual-profile pair)
Paper reference: Nebby §3.4 steps 4 & 5  (QUIC adaptation)

HOW THIS RELATES TO YOUR EXISTING CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your existing pipeline (classify.py):
    compute_bif()  →  smooth_bif()  →  remove_slow_start()
    →  segment_bif()  →  extract_features()  →  GNB / BBR rule

This file is identical EXCEPT for the first step:
    compute_bif_quic()  →  smooth_bif()  →  remove_slow_start()
    →  segment_bif()   →  extract_features()  →  GNB / BBR rule

Everything from smooth_bif() onwards is 100% reused from your
existing bif.py, preprocess.py, features.py, and classify.py.
The GNB model (gnb.pkl) is also shared — QUIC CCAs produce the
same polynomial BiF shapes as their TCP counterparts in the same
CCA family (CUBIC, NewReno, etc.).

PRIMARY FUNCTIONS
━━━━━━━━━━━━━━━━
  classify_quic_pair(csv_50ms, csv_100ms)  ← preferred (6D, dual profile)
  classify_quic_single(csv_path)           ← fallback  (3D, one profile)

BBR DETECTION FOR QUIC
━━━━━━━━━━━━━━━━━━━━━━
The same detect_bbr() from classify.py works on QUIC BiF traces
because the ProbeRTT / ProbeBW signatures are properties of the
CCA's rate-control logic, not of the transport protocol.

Usage:
    # Dual profile (preferred — matches training):
    python3 classify_quic.py  <csv_50ms_quic>  <csv_100ms_quic>

    # Single profile fallback:
    python3 classify_quic.py  <any_quic_csv>
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))

# ── your existing modules (unchanged) ────────────────────────────────────────
from bif        import smooth_bif              # reused as-is
from preprocess import remove_slow_start, segment_bif   # reused as-is
from features   import (extract_features,
                         extract_features_dual_profile)  # reused as-is
from classify   import detect_bbr, _majority_vote        # reused as-is

# ── new QUIC-only module ──────────────────────────────────────────────────────
from quic_bif   import compute_bif_quic

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR = '../models'
SERVER_IP = None       # None = auto-detect from CSV
RTT_S     = 0.1        # seconds (fallback for single-profile)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    gnb_path = os.path.join(MODEL_DIR, 'gnb.pkl')
    le_path  = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if not os.path.exists(gnb_path) or not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Model not found in {MODEL_DIR}.\n"
            "Run  python3 train.py  first."
        )
    return joblib.load(gnb_path), joblib.load(le_path)


def _quic_pipeline(csv_path, rtt_s, server_ip=None):
    """
    Full preprocessing pipeline for one QUIC CSV.
    Returns (t_ss, bif_ss, segments, features_3d).
    """
    t, bif         = compute_bif_quic(csv_path, server_ip)   # ← only QUIC change
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)               # identical to TCP
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)           # identical to TCP
    segments       = segment_bif(t_ss, bif_ss)               # identical to TCP
    feats          = extract_features(segments)               # identical to TCP
    return t_ss, bif_ss, segments, feats


def _check_bbr_quic(csv_path, server_ip=None, rtt_s=RTT_S):
    """Run BBR rule detector on one QUIC CSV. Returns 'bbr' or None."""
    t, bif         = compute_bif_quic(csv_path, server_ip)
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
    return detect_bbr(t_ss, bif_ss, rtt_s), t_ss, bif_ss


def _get_quic_features(csv_path, rtt_s, server_ip=None):
    """
    Load one QUIC CSV → 3D feature array  (n_segments, 3).
    Mirrors _get_features_from_csv() in features.py but uses
    compute_bif_quic() instead of compute_bif().
    """
    t, bif         = compute_bif_quic(csv_path, server_ip)
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
    segments       = segment_bif(t_ss, bif_ss)
    return extract_features(segments)   # (n_segs, 3)


def _dual_profile_quic(csv_50ms, csv_100ms, server_ip=None):
    """
    Extract 6D features from a QUIC dual-profile pair.
    Mirrors extract_features_dual_profile() in features.py but uses
    compute_bif_quic() for both profiles.

    Returns (feats_6d, n_paired_segments).
    """
    f50  = _get_quic_features(csv_50ms,  rtt_s=0.10, server_ip=server_ip)
    f100 = _get_quic_features(csv_100ms, rtt_s=0.20, server_ip=server_ip)

    n = min(len(f50), len(f100))
    if n == 0:
        return np.empty((0, 6)), 0

    feats_6d = np.hstack([f50[:n], f100[:n]])   # (n, 6)
    return feats_6d, n


# ─────────────────────────────────────────────────────────────────────────────
# Primary: dual-profile QUIC classification  (6D — matches training)
# ─────────────────────────────────────────────────────────────────────────────

def classify_quic_pair(csv_50ms, csv_100ms,
                        server_ip=SERVER_IP, verbose=True):
    """
    Classify a QUIC flow using BOTH delay profiles — 6D feature vector.
    This is the correct function when you have both profiles.

    Steps:
      1. Run BBR rule detector on the 50ms trace.
      2. If not BBR, extract 6D dual-profile features.
      3. Run GNB → majority vote across segments.

    Parameters
    ----------
    csv_50ms  : path to QUIC CSV from the 50ms delay run
    csv_100ms : path to QUIC CSV from the 100ms delay run
    server_ip : IP of the QUIC server (None = auto-detect)

    Returns
    -------
    label      : str   — predicted CCA (e.g. 'cubic', 'bbr', 'reno')
    confidence : float — fraction of segments agreeing on this label
    """
    gnb, le = load_model()

    # Step 1: BBR rule check (works on either profile)
    bbr, t_ss, bif_ss = _check_bbr_quic(csv_50ms, server_ip, rtt_s=0.10)
    if bbr:
        if verbose:
            print("  Result : BBR  (rule-based — ProbeBW/ProbeRTT pattern "
                  "detected in QUIC BiF)")
        return 'bbr', 1.0

    # Step 2: dual-profile 6D features
    feats, n = _dual_profile_quic(csv_50ms, csv_100ms, server_ip)

    if n == 0:
        if verbose:
            print("  Result : Unknown  (no usable segments in QUIC trace)")
        return 'unknown', 0.0

    # Step 3: GNB
    n_model = gnb.theta_.shape[1]
    if n_model == 6:
        pass   # correct — model was trained on 6D features
    elif n_model == 3:
        # Model is 3D — use only 50ms features
        if verbose:
            print("  WARNING: Model is 3D but QUIC dual-profile gives 6D. "
                  "Using only 50ms features.")
        feats = feats[:, :3]

    preds          = gnb.predict(feats)
    label, conf    = _majority_vote(preds, le)

    if verbose:
        print(f"  Result : {label.upper()}  "
              f"(confidence: {conf:.0%},  {n} segment pairs,  "
              f"6D dual-profile QUIC)")
        unique, counts = np.unique(preds, return_counts=True)
        for enc, cnt in zip(unique, counts):
            cls = le.inverse_transform([enc])[0]
            print(f"    {cls:<12} {cnt:>3}  {'█' * cnt}")

    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: single-profile QUIC classification  (3D)
# ─────────────────────────────────────────────────────────────────────────────

def classify_quic_single(csv_path, server_ip=SERVER_IP,
                          rtt_s=RTT_S, verbose=True):
    """
    Classify a single QUIC CSV — 3D feature vector.
    Lower accuracy than classify_quic_pair(); use when you only
    have one profile.

    Parameters
    ----------
    csv_path  : path to any QUIC UDP CSV
    server_ip : IP of the QUIC server (None = auto-detect)
    rtt_s     : assumed RTT in seconds (0.10 for 50ms delay,
                0.20 for 100ms delay)

    Returns
    -------
    label      : str
    confidence : float
    """
    gnb, le = load_model()

    # BBR check
    bbr, t_ss, bif_ss = _check_bbr_quic(csv_path, server_ip, rtt_s)
    if bbr:
        if verbose:
            print("  Result : BBR  (rule-based)")
        return 'bbr', 1.0

    # Single-profile features
    feats = _get_quic_features(csv_path, rtt_s, server_ip)

    if len(feats) == 0:
        if verbose:
            print("  Result : Unknown  (no segments)")
        return 'unknown', 0.0

    n_model = gnb.theta_.shape[1]
    if n_model == 6:
        if verbose:
            print("  WARNING: Model trained on 6D but only 1 profile provided. "
                  "Padding 3D → 6D (accuracy will be lower).")
        feats = np.hstack([feats, feats])   # crude pad — avoids crash

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)

    if verbose:
        print(f"  Result : {label.upper()}  "
              f"(confidence: {conf:.0%},  {len(feats)} segments,  "
              f"{'3D' if n_model==3 else '3D→6D padded'})")
    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) == 3:
        csv50, csv100 = sys.argv[1], sys.argv[2]
        for p in [csv50, csv100]:
            if not os.path.exists(p):
                print(f"File not found: {p}")
                sys.exit(1)
        print(f"\nClassifying QUIC (dual profile):")
        print(f"  50ms : {csv50}")
        print(f"  100ms: {csv100}")
        print("-" * 60)
        label, conf = classify_quic_pair(csv50, csv100)

    elif len(sys.argv) == 2:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            sys.exit(1)
        print(f"\nClassifying QUIC (single profile — lower accuracy):")
        print(f"  {csv_path}")
        print("-" * 60)
        label, conf = classify_quic_single(csv_path)

    else:
        print("Usage:")
        print("  python3 classify_quic.py <csv_50ms> <csv_100ms>   ← preferred")
        print("  python3 classify_quic.py <any_quic_csv>           ← fallback")
        sys.exit(1)

    print("-" * 60)
    print(f"Final answer: {label.upper()}  (confidence: {conf:.0%})\n")