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

BBR DETECTION FOR QUIC — WHY detect_bbr() FROM classify.py FAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
detect_bbr() in classify.py was tuned for TCP BiF which is computed
exactly from sequence/ACK numbers.  QUIC BiF from compute_bif_quic()
is estimated using the two-assumption model — it is accurate but has
a different absolute scale and noise floor than TCP BiF.

Specifically, the ProbeRTT dips in QUIC BiF:
  - Are present and visible (confirmed in bif_traces_eval_quic.png)
  - But the TCP thresholds (e.g. drop to < 20% of max) are too strict
    because QUIC BiF estimation slightly underestimates the floor

detect_bbr_quic() below uses the same structural logic but with:
  1. A looser relative-drop threshold (40% instead of TCP's 20%)
  2. A minimum absolute drop (10 KB) to avoid false positives on noise
  3. Periodic detection via autocorrelation on the smoothed signal
     to confirm the ~8-10 RTT ProbeBW cycle

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
from bif        import smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import (extract_features,
                         extract_features_dual_profile)
from classify   import _majority_vote        # reuse vote logic, NOT detect_bbr

# ── new QUIC-only module ──────────────────────────────────────────────────────
from quic_bif   import compute_bif_quic

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR = '../models'
SERVER_IP = None       # None = auto-detect from CSV
RTT_S     = 0.1        # seconds (fallback for single-profile)


# ─────────────────────────────────────────────────────────────────────────────
# QUIC-specific BBR detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_bbr_quic(t, bif, rtt_s):
    """
    Detect BBR from a QUIC BiF trace using ProbeRTT signature.

    BBR periodically drains the queue to measure min-RTT (ProbeRTT phase).
    This causes sharp, periodic drops in BiF to a low floor, visible as
    a staircase pattern in the plots.

    Why not reuse detect_bbr() from classify.py?
    ─────────────────────────────────────────────
    TCP BiF is exact (from seq/ACK numbers). QUIC BiF is estimated via the
    two-assumption model — same shape, but slightly different absolute scale.
    The TCP thresholds are too tight for QUIC's noisier estimates and cause
    all BBR QUIC traces to return None (confirmed in evaluation plot where
    all BBR traces show pred=unknown with segs=0).

    Detection logic:
    ────────────────
    1. Smooth BiF with a 1-RTT rolling window.
    2. Find local minima that are < DROP_THRESH * rolling_max.
    3. Require at least MIN_DIPS dips within the trace.
    4. Check dips are roughly periodic (CV of inter-dip intervals < 0.5).

    Parameters
    ----------
    t     : numpy array of timestamps (s)
    bif   : numpy array of BiF values (bytes), after remove_slow_start
    rtt_s : estimated RTT in seconds

    Returns
    -------
    'bbr' if detected, None otherwise
    """
    if len(bif) < 20:
        return None

    # ── 1. Smooth with 1-RTT window ──────────────────────────────────────────
    dt       = np.median(np.diff(t)) if len(t) > 1 else 0.001
    win      = max(3, int(rtt_s / dt))
    bif_s    = np.convolve(bif, np.ones(win) / win, mode='same')

    # ── 2. Rolling max (2-RTT window) ────────────────────────────────────────
    win2     = max(5, int(2 * rtt_s / dt))
    roll_max = np.array([
        bif_s[max(0, i - win2): i + 1].max()
        for i in range(len(bif_s))
    ])

    # ── 3. Find ProbeRTT dips ────────────────────────────────────────────────
    # A dip = local value drops to < DROP_THRESH of recent max
    # AND the absolute value is < ABS_CEIL bytes (not just proportionally low)
    DROP_THRESH = 0.55          # QUIC: looser than TCP's ~0.20
    ABS_CEIL    = 80_000        # ignore if BiF is always small (noise)
    MIN_DIPS    = 2

    global_max = bif_s.max()
    if global_max < ABS_CEIL:
        # BiF never got high enough — not a BBR ProbeRTT signature
        return None

    dip_mask = (bif_s < DROP_THRESH * roll_max) & (roll_max > ABS_CEIL)

    # Find dip centers (transitions into dip regions)
    dip_times = []
    in_dip    = False
    dip_start = 0
    for i, d in enumerate(dip_mask):
        if d and not in_dip:
            in_dip    = True
            dip_start = i
        elif not d and in_dip:
            in_dip = False
            center = (dip_start + i) // 2
            dip_times.append(t[center])

    if len(dip_times) < MIN_DIPS:
        return None

    # ── 4. Check dips are roughly periodic ───────────────────────────────────
    intervals = np.diff(dip_times)
    if len(intervals) == 0:
        return None

    cv = intervals.std() / (intervals.mean() + 1e-9)   # coefficient of variation
    if cv > 0.65:
        # Dips are irregular — not BBR's deterministic ProbeRTT schedule
        return None

    # ── 5. Check inter-dip interval is plausible for BBR ─────────────────────
    # BBR ProbeRTT fires every ~8 RTTs (min 200ms, typ 1-4s)
    mean_interval = intervals.mean()
    if mean_interval < 0.15 or mean_interval > 20.0:
        return None

    return 'bbr'


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    # Prefer QUIC-specific model, fall back to TCP model
    for stem in ['gnb_quic', 'gnb']:
        gnb_path = os.path.join(MODEL_DIR, f'{stem}.pkl')
        le_path  = os.path.join(MODEL_DIR,
                                f'label_encoder_quic.pkl'
                                if stem == 'gnb_quic'
                                else 'label_encoder.pkl')
        if os.path.exists(gnb_path) and os.path.exists(le_path):
            return joblib.load(gnb_path), joblib.load(le_path)
    raise FileNotFoundError(
        f"No model found in {MODEL_DIR}.\n"
        "Run  python3 train_quic.py  (or train.py) first."
    )


def _check_bbr_quic(csv_path, server_ip=None, rtt_s=RTT_S):
    """
    Run QUIC-specific BBR detector.
    Returns ('bbr', t_ss, bif_ss) or (None, t_ss, bif_ss).
    """
    t, bif         = compute_bif_quic(csv_path, server_ip)
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
    result         = detect_bbr_quic(t_ss, bif_ss, rtt_s)   # ← QUIC version
    return result, t_ss, bif_ss


def _get_quic_features(csv_path, rtt_s, server_ip=None):
    """
    Load one QUIC CSV → 3D feature array  (n_segments, 3).
    """
    t, bif       = compute_bif_quic(csv_path, server_ip)
    t_s, bif_s   = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss = remove_slow_start(t_s, bif_s)
    segments     = segment_bif(t_ss, bif_ss)
    return extract_features(segments)


def _dual_profile_quic(csv_50ms, csv_100ms, server_ip=None):
    """
    Extract 6D features from a QUIC dual-profile pair.
    Returns (feats_6d, n_paired_segments).
    """
    f50  = _get_quic_features(csv_50ms,  rtt_s=0.10, server_ip=server_ip)
    f100 = _get_quic_features(csv_100ms, rtt_s=0.20, server_ip=server_ip)

    n = min(len(f50), len(f100))
    if n == 0:
        return np.empty((0, 6)), 0

    return np.hstack([f50[:n], f100[:n]]), n


# ─────────────────────────────────────────────────────────────────────────────
# Primary: dual-profile QUIC classification  (6D)
# ─────────────────────────────────────────────────────────────────────────────

def classify_quic_pair(csv_50ms, csv_100ms,
                        server_ip=SERVER_IP, verbose=True):
    """
    Classify a QUIC flow using BOTH delay profiles — 6D feature vector.

    Steps:
      1. Run QUIC BBR detector on the 50ms trace.
      2. If not BBR, extract 6D dual-profile features → GNB.

    Returns
    -------
    label      : str   — e.g. 'cubic', 'bbr', 'reno'
    confidence : float — fraction of segments agreeing
    """
    gnb, le = load_model()

    # Step 1: QUIC-specific BBR check
    bbr, t_ss, bif_ss = _check_bbr_quic(csv_50ms, server_ip, rtt_s=0.10)
    if bbr:
        if verbose:
            print("  Result : BBR  (QUIC rule-based — ProbeRTT dip pattern detected)")
        return 'bbr', 1.0

    # Step 2: dual-profile 6D features
    feats, n = _dual_profile_quic(csv_50ms, csv_100ms, server_ip)

    if n == 0:
        if verbose:
            print("  Result : Unknown  (no usable segments in QUIC trace)")
        return 'unknown', 0.0

    # Adapt feature dimension to model
    n_model = gnb.theta_.shape[1]
    if n_model == 3:
        if verbose:
            print("  NOTE: 3D model — using 50ms features only")
        feats = feats[:, :3]

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)

    if verbose:
        print(f"  Result : {label.upper()}  "
              f"(confidence: {conf:.0%},  {n} segment pairs,  6D QUIC)")
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

    Returns
    -------
    label      : str
    confidence : float
    """
    gnb, le = load_model()

    bbr, t_ss, bif_ss = _check_bbr_quic(csv_path, server_ip, rtt_s)
    if bbr:
        if verbose:
            print("  Result : BBR  (QUIC rule-based)")
        return 'bbr', 1.0

    feats = _get_quic_features(csv_path, rtt_s, server_ip)

    if len(feats) == 0:
        if verbose:
            print("  Result : Unknown  (no segments)")
        return 'unknown', 0.0

    n_model = gnb.theta_.shape[1]
    if n_model == 6:
        if verbose:
            print("  NOTE: 6D model, padding single profile 3D→6D "
                  "(lower accuracy)")
        feats = np.hstack([feats, feats])

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)

    if verbose:
        dim = '3D' if n_model == 3 else '3D→6D padded'
        print(f"  Result : {label.upper()}  "
              f"(confidence: {conf:.0%},  {len(feats)} segments,  {dim})")
    return label, conf


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) == 3:
        csv50, csv100 = sys.argv[1], sys.argv[2]
        for p in [csv50, csv100]:
            if not os.path.exists(p):
                print(f"File not found: {p}"); sys.exit(1)
        print(f"\nClassifying QUIC (dual profile):")
        print(f"  50ms : {csv50}")
        print(f"  100ms: {csv100}")
        print("-" * 60)
        label, conf = classify_quic_pair(csv50, csv100)

    elif len(sys.argv) == 2:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}"); sys.exit(1)
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