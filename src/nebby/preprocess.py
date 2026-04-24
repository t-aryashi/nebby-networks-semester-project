"""
preprocess.py — Slow-start removal and trace segmentation
Paper reference: Nebby §3.4 step 2 (Segmentation)
"""

import numpy as np


def remove_slow_start(t, bif, drop_fraction=0.4):
    """
    Discard the slow-start phase.

    Slow start looks the same for every CCA so the paper ignores it.
    We detect its end as the first large BiF drop (>40%) which signals
    the first buffer overflow / congestion event.

    Parameters
    ----------
    t             : timestamps from smooth_bif
    bif           : smoothed BiF values
    drop_fraction : fraction of drop that counts as a loss event

    Returns
    -------
    t_ca, bif_ca  : trace starting from congestion-avoidance phase
    """
    for i in range(10, len(bif)):
        if bif[i - 1] > 500 and bif[i] < bif[i - 1] * (1.0 - drop_fraction):
            print(f"  Slow start ends at t={t[i]:.2f}s  "
                  f"(BiF {bif[i-1]/1024:.1f}→{bif[i]/1024:.1f} KB)")
            return t[i:], bif[i:]

    # Fallback: skip first 15% of the trace
    cut = max(1, int(0.15 * len(t)))
    print(f"  No clear slow-start end found — skipping first {t[cut]:.1f}s")
    return t[cut:], bif[cut:]


def segment_bif(t, bif, drop_fraction=0.35,
                min_duration_s=1.0, min_points=20):
    """
    Split the congestion-avoidance trace into individual segments.

    Each segment is one oscillation cycle — the region between two
    consecutive back-offs. A back-off is a drop of >35% in BiF which
    indicates a loss event (CUBIC/Reno) or a ProbeRTT phase (BBR).

    Parameters
    ----------
    t               : timestamps (congestion-avoidance phase)
    bif             : BiF values  (congestion-avoidance phase)
    drop_fraction   : minimum drop fraction to count as a back-off
    min_duration_s  : discard segments shorter than this (seconds)
    min_points      : discard segments with fewer points than this

    Returns
    -------
    segments : list of (t_seg, bif_seg) tuples
    """
    segments  = []
    seg_start = 0

    for i in range(5, len(bif)):
        is_backoff = (
            bif[i - 1] > 500                                   # meaningful level
            and bif[i] < bif[i - 1] * (1.0 - drop_fraction)   # big drop
        )
        if is_backoff:
            seg_t   = t[seg_start:i]
            seg_bif = bif[seg_start:i]
            duration = seg_t[-1] - seg_t[0] if len(seg_t) > 1 else 0

            if duration >= min_duration_s and len(seg_bif) >= min_points:
                segments.append((seg_t, seg_bif))

            seg_start = i   # next segment starts after the back-off

    # Include the last segment (no trailing back-off)
    if len(t) - seg_start >= min_points:
        seg_t   = t[seg_start:]
        seg_bif = bif[seg_start:]
        if len(seg_bif) >= min_points:
            segments.append((seg_t, seg_bif))

    print(f"  Found {len(segments)} segments")
    return segments