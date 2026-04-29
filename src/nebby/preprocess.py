"""
preprocess.py — Slow-start removal and trace segmentation
Paper reference: Nebby §3.4 step 2 (Segmentation)

CHANGES FOR 2000 Kbps:
  - BIF_MIN_BYTES raised from 500 → 5000
      At 200 Kbps, BiF peaks at ~5 KB so 500 bytes was meaningful.
      At 2000 Kbps, BiF peaks at ~50 KB so 500 bytes is effectively
      always true and provides no filtering. 5000 bytes (5 KB) is
      about 10% of the minimum expected BiF at 2000 Kbps — a sensible
      floor that filters out transient startup noise.

  - min_duration_s default lowered from 1.0 → 0.5 seconds
      At higher bandwidth, oscillation periods are shorter because
      cwnd fills faster. CUBIC at 2000 Kbps completes one sawtooth
      cycle in ~2-4 seconds vs ~5-15 seconds at 200 Kbps.
      Keeping min_duration_s=1.0 would discard valid short segments.

Everything else (drop_fraction, min_points) unchanged — these are
relative/count-based so they scale automatically with bandwidth.
"""

import numpy as np

# ── tunable thresholds ────────────────────────────────────────────────────────
# Raise this if you change bandwidth. Rule of thumb:
#   BIF_MIN_BYTES ≈ BDP / 5   where BDP = BW_bytes/s × RTT_s
#   200 Kbps, 100ms RTT:  BDP = 2500 bytes  → min ≈  500
#   2000 Kbps, 100ms RTT: BDP = 25000 bytes → min ≈ 5000
BIF_MIN_BYTES = 5000   # bytes — ignore BiF values below this


def remove_slow_start(t, bif, drop_fraction=0.4):
    """
    Discard the slow-start phase.

    Slow start looks the same for every CCA — exponential growth
    followed by the first buffer overflow. The paper ignores it.

    We detect the end of slow start as the first large BiF drop
    (> drop_fraction of the current value) above BIF_MIN_BYTES.

    Parameters
    ----------
    t             : timestamps from smooth_bif
    bif           : smoothed BiF values (bytes)
    drop_fraction : fraction drop that counts as first loss event

    Returns
    -------
    t_ca, bif_ca  : trace from the congestion-avoidance phase onward
    """
    for i in range(10, len(bif)):
        if (bif[i - 1] > BIF_MIN_BYTES and
                bif[i] < bif[i - 1] * (1.0 - drop_fraction)):
            print(f"  Slow start ends at t={t[i]:.2f}s  "
                  f"(BiF {bif[i-1]/1024:.1f} → {bif[i]/1024:.1f} KB)")
            return t[i:], bif[i:]

    # Fallback: skip first 15% of trace
    cut = max(1, int(0.15 * len(t)))
    print(f"  No clear slow-start end — skipping first {t[cut]:.1f}s")
    return t[cut:], bif[cut:]


def segment_bif(t, bif,
                drop_fraction=0.35,
                min_duration_s=0.5,
                min_points=20,
                bif_min=None):
    """
    Split the congestion-avoidance trace into individual segments.

    Each segment is one oscillation cycle — the region between two
    consecutive back-offs. A back-off is detected as a drop of more
    than drop_fraction in BiF, occurring above bif_min.

    Parameters
    ----------
    t               : timestamps (congestion-avoidance phase)
    bif             : BiF values  (congestion-avoidance phase)
    drop_fraction   : minimum fractional drop to count as a back-off
    min_duration_s  : discard segments shorter than this (seconds)
    min_points      : discard segments with fewer points than this
    bif_min         : minimum BiF floor to count a drop as a back-off.
                      Defaults to BIF_MIN_BYTES (global, tuned for bulk
                      transfers). Pass a lower value for short browser
                      flows where per-flow BiF is naturally lower due
                      to bandwidth sharing between concurrent connections.

    Returns
    -------
    segments : list of (t_seg, bif_seg) tuples
    """
    floor     = bif_min if bif_min is not None else BIF_MIN_BYTES
    segments  = []
    seg_start = 0

    for i in range(5, len(bif)):
        is_backoff = (
            bif[i - 1] > floor and
            bif[i] < bif[i - 1] * (1.0 - drop_fraction)
        )
        if is_backoff:
            seg_t    = t[seg_start:i]
            seg_bif  = bif[seg_start:i]
            duration = seg_t[-1] - seg_t[0] if len(seg_t) > 1 else 0

            if duration >= min_duration_s and len(seg_bif) >= min_points:
                segments.append((seg_t, seg_bif))

            seg_start = i   # next segment starts after the back-off

    # Last segment (no trailing back-off)
    if len(t) - seg_start >= min_points:
        seg_t   = t[seg_start:]
        seg_bif = bif[seg_start:]
        if len(seg_bif) >= min_points:
            segments.append((seg_t, seg_bif))

    print(f"  Found {len(segments)} segments  "
          f"(min_duration={min_duration_s}s, min_pts={min_points}, "
          f"BiF_min={floor}B)")
    return segments