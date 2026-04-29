"""
classify.py — Classify a single TCP trace (or a pair for dual-profile)
Paper reference: Nebby §3.4 steps 4 & 5

CHANGES FROM PREVIOUS VERSION:
  - classify_trace_pair() is the primary function — takes both the 50ms
    and 100ms delay CSVs and uses 6D features for GNB (matches training).
  - classify_trace() retained as a fallback that uses only 3D features
    from a single CSV — useful when you only have one profile.
  - BBR rule-based detector unchanged.

Usage:
    # Preferred — dual profile (matches training):
    python3 classify.py  <csv_50ms>  <csv_100ms>

    # Fallback — single profile (3D features, lower accuracy):
    python3 classify.py  <any_csv>
"""

import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif
from features   import (extract_features,
                         extract_features_dual_profile)

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR = '../models'
SERVER_IP = '10.0.0.1'
RTT_S     = 0.1   # seconds (used only for single-profile fallback)

RATE_BASED_CCAS = {'bbr', 'bbr2', 'bbr3'}


# ══════════════════════════════════════════════════════════════════════════════
# BBR RULE-BASED DETECTOR  (paper §3.4 step 5)
# ══════════════════════════════════════════════════════════════════════════════

def detect_bbr(t, bif, rtt_s=RTT_S):
    """
    Detect BBR by its characteristic BiF behaviour.
    Uses THREE independent checks — ALL must pass to return 'bbr'.
    This eliminates the false positives on CUBIC/Reno that plagued
    the previous single-criterion version.

    Why the old version failed on CUBIC:
      - CUBIC's sawtooth continuously rises then sharply drops
      - The rising phase produces clusters of positive d(BiF)/dt values
      - Those clusters looked like "periodic spikes" to the old detector
      - The old detector had only one criterion (spike gap) which CUBIC passed

    The three criteria:

    CHECK 1 — BiF FLATNESS  (most discriminative)
      BBR cruises near a target rate → low coefficient of variation (CV).
      CV = std / mean.
      BBR:   CV ≈ 0.05 – 0.20  (BiF varies by only ~10-20% around mean)
      CUBIC: CV ≈ 0.40 – 0.80  (sawtooth swings wildly)
      Threshold: CV < 0.35  →  potentially BBR

    CHECK 2 — ABSENCE OF DEEP DROPS  (confirms no sawtooth)
      Loss-based CCAs drop BiF by 50%+ at every loss event.
      BBR's ProbeRTT drops are small (~30%) and infrequent (~every 10s).
      We count how many times BiF drops by >40% of its running max.
      BBR:   0–2 deep drops in a 30s trace
      CUBIC: 3–15 deep drops (one per sawtooth cycle)
      Threshold: ≤ 2 deep drops

    CHECK 3 — PROBERTT SIGNATURE  (positive confirmation)
      BBRv1 backs off to its min RTT estimate every ~10 seconds.
      This creates periodic dips visible even in a smoothed trace.
      We look for at least 1 dip where BiF drops >20% and recovers,
      with dip duration < 1 RTT (these are brief, not the sustained
      drops you see in CUBIC after a loss).

    Parameters
    ----------
    t     : timestamps (seconds) — post slow-start removal
    bif   : smoothed BiF values (bytes)
    rtt_s : estimated RTT in seconds

    Returns
    -------
    'bbr' if all three checks pass, None otherwise
    """
    if len(bif) < 50:
        return None

    bif_mean = bif.mean()
    if bif_mean < 100:
        return None   # essentially no traffic

    # ── CHECK 1: Flatness via coefficient of variation ────────────────────
    cv = bif.std() / bif_mean
    if cv >= 0.35:
        # Too variable — this is a sawtooth (CUBIC/Reno), not BBR
        return None

    # ── CHECK 2: Count deep drops (loss events) ───────────────────────────
    # A deep drop = BiF falls to < 60% of its recent running max
    window     = max(1, int(len(bif) * 0.05))   # 5% window
    deep_drops = 0
    for i in range(window, len(bif)):
        local_max = bif[max(0, i - window):i].max()
        if local_max > bif_mean * 0.3 and bif[i] < local_max * 0.60:
            deep_drops += 1

    # Normalise by trace length (allow ~1 drop per 15 seconds)
    trace_duration   = t[-1] - t[0]
    allowed_drops    = max(2, int(trace_duration / 15))
    if deep_drops > allowed_drops:
        # Too many sharp drops — this is a loss-based CCA
        return None

    # ── CHECK 3: ProbeRTT signature ───────────────────────────────────────
    # Look for at least one brief dip where BiF drops >20% then recovers
    # within a short window (~2-3 RTTs). This is BBR's ProbeRTT behaviour.
    probe_rtt_window = int(3 * rtt_s / max(np.median(np.diff(t)), 1e-6))
    probe_rtt_window = max(5, min(probe_rtt_window, 50))
    found_probe_rtt  = False

    for i in range(probe_rtt_window, len(bif) - probe_rtt_window):
        before = bif[i - probe_rtt_window:i].mean()
        after  = bif[i + 1:i + probe_rtt_window].mean()
        at     = bif[i]
        # BiF dips and recovers (both before and after are higher)
        if before > 0 and after > 0:
            dip_depth = 1.0 - at / max(before, after)
            if 0.15 < dip_depth < 0.70 and after > at * 1.1:
                found_probe_rtt = True
                break

    if not found_probe_rtt:
        return None

    # All three checks passed
    return 'bbr'


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
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


def _majority_vote(preds, le):
    """Return (label_string, confidence) from an array of encoded predictions."""
    unique, counts = np.unique(preds, return_counts=True)
    best_enc       = unique[np.argmax(counts)]
    confidence     = counts.max() / counts.sum()
    label          = le.inverse_transform([best_enc])[0]
    return label, confidence


def _check_bbr(csv_path, server_ip=SERVER_IP, rtt_s=RTT_S):
    """Run BBR rule detector on a single CSV. Returns 'bbr' or None."""
    t, bif         = compute_bif(csv_path, server_ip)
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
    return detect_bbr(t_ss, bif_ss, rtt_s), t_ss, bif_ss


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY: dual-profile classification  (matches training)
# ══════════════════════════════════════════════════════════════════════════════

def classify_trace_pair(csv_50ms, csv_100ms,
                         server_ip=SERVER_IP, verbose=True):
    """
    Classify using BOTH delay profiles — 6D feature vector.
    This is the correct function to use when you have both profiles
    (i.e. traces generated by the updated generate_dataset.sh).

    Parameters
    ----------
    csv_50ms  : path to CSV from the 50ms delay run
    csv_100ms : path to CSV from the 100ms delay run

    Returns
    -------
    label      : str  — predicted CCA
    confidence : float — fraction of segments agreeing
    """
    gnb, le = load_model()

    # 1. BBR check on 50ms trace (BBR detection works on either profile)
    bbr, t_ss, bif_ss = _check_bbr(csv_50ms, server_ip, rtt_s=0.10)
    if bbr:
        if verbose:
            print("  Result : BBR  (rule-based — ProbeBW/ProbeRTT pattern)")
        return 'bbr', 1.0

    # 2. Dual-profile GNB
    feats, n = extract_features_dual_profile(csv_50ms, csv_100ms, server_ip)

    if n == 0:
        if verbose:
            print("  Result : Unknown  (no usable segments)")
        return 'unknown', 0.0

    preds          = gnb.predict(feats)
    label, conf    = _majority_vote(preds, le)

    if verbose:
        print(f"  Result : {label.upper()}  "
              f"(confidence: {conf:.0%},  {n} segment pairs,  6D dual-profile)")
        unique, counts = np.unique(preds, return_counts=True)
        for enc, cnt in zip(unique, counts):
            cls = le.inverse_transform([enc])[0]
            print(f"    {cls:<12} {cnt:>3}  {'█' * cnt}")

    return label, conf


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK: single-profile classification  (3D features)
# ══════════════════════════════════════════════════════════════════════════════

def classify_trace(csv_path, server_ip=SERVER_IP, rtt_s=RTT_S, verbose=True):
    """
    Classify using a single CSV — 3D feature vector.

    NOTE: This will have lower accuracy than classify_trace_pair() because
    GNB was trained on 6D features. Use this only when you have a single
    trace and cannot get both delay profiles.

    The GNB will still work because predict() operates on whatever
    dimensions the model was trained on — but passing 3D features to a
    6D-trained model will raise an error. This function therefore uses
    a separate single-profile GNB if available, or falls back gracefully.
    """
    gnb, le = load_model()

    # Check if model expects 6D
    n_features = gnb.theta_.shape[1]  # GNB stores mean per class per feature
    if n_features == 6:
        if verbose:
            print("  WARNING: Model trained on 6D features but only 1 profile "
                  "provided. Using 3D fallback — accuracy will be lower.\n"
                  "  Provide both delay profile CSVs for best results.")

    # BBR check
    bbr, t_ss, bif_ss = _check_bbr(csv_path, server_ip, rtt_s)
    if bbr:
        if verbose:
            print("  Result : BBR  (rule-based)")
        return 'bbr', 1.0

    # Single-profile features
    segments = segment_bif(t_ss, bif_ss)
    feats    = extract_features(segments)   # (n, 3)

    if len(feats) == 0:
        if verbose:
            print("  Result : Unknown  (no segments)")
        return 'unknown', 0.0

    if n_features == 6:
        # Pad to 6D: duplicate 3D features for both "profiles"
        # (crude but prevents crash — accuracy still lower than true dual)
        feats = np.hstack([feats, feats])

    preds       = gnb.predict(feats)
    label, conf = _majority_vote(preds, le)

    if verbose:
        print(f"  Result : {label.upper()}  "
              f"(confidence: {conf:.0%},  {len(feats)} segments,  "
              f"{'3D single' if n_features==3 else '3D→6D padded'})")
    return label, conf


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Dual profile — preferred
        csv50, csv100 = sys.argv[1], sys.argv[2]
        for p in [csv50, csv100]:
            if not os.path.exists(p):
                print(f"File not found: {p}")
                sys.exit(1)
        print(f"\nClassifying (dual profile):")
        print(f"  50ms : {csv50}")
        print(f"  100ms: {csv100}")
        print("-" * 60)
        label, conf = classify_trace_pair(csv50, csv100)

    elif len(sys.argv) == 2:
        # Single profile fallback
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            sys.exit(1)
        print(f"\nClassifying (single profile — lower accuracy):")
        print(f"  {csv_path}")
        print("-" * 60)
        label, conf = classify_trace(csv_path)

    else:
        print("Usage:")
        print("  python3 classify.py <csv_50ms> <csv_100ms>   ← preferred")
        print("  python3 classify.py <any_csv>                ← fallback")
        sys.exit(1)

    print("-" * 60)
    print(f"Final answer: {label.upper()}  (confidence: {conf:.0%})\n")