"""
features.py — Polynomial feature extraction from BiF segments
Paper reference: Nebby §3.4 steps 3 & 4

CHANGES FROM PREVIOUS VERSION:
  - Added extract_features_dual_profile() which concatenates features
    from both the 50ms and 100ms delay profiles into a 6D vector
    [a1,b1,c1, a2,b2,c2]. This is how the paper actually disambiguates
    CCAs that look similar at one delay but different at another
    (e.g. New Reno vs Illinois vs HSTCP look the same at 50ms but
    differ at 100ms — paper Table 2 and §3.3).
  - Single-profile extract_features() retained for classify.py fallback.

Feature vector structure:
  Single profile : [a, b, c]              shape (n_segments, 3)
  Dual profile   : [a1,b1,c1, a2,b2,c2]  shape (n_segments, 6)
  where subscript 1 = 50ms profile, 2 = 100ms profile
"""

import numpy as np

from bif        import compute_bif, smooth_bif
from preprocess import remove_slow_start, segment_bif


# ── polynomial fitting constants (paper §3.4) ─────────────────────────────────
LAMBDA_PENALTY = 0.7   # regularisation weight — paper empirically set to 0.7
N_SAMPLE_PTS   = 200   # number of points to sample per segment


def fit_segment(seg_bif, lambda_penalty=LAMBDA_PENALTY):
    """
    Fit a normalised BiF segment with polynomials of degree 1 / 2 / 3.
    Pick the best fit using the penalised MSE from the paper:

        Error = MSE + λ × Degree × Σ|coefficients|

    Parameters
    ----------
    seg_bif        : 1-D array of BiF values for one segment
    lambda_penalty : regularisation weight (paper uses 0.7)

    Returns
    -------
    numpy array of shape (3,) → [a, b, c]
    None if the segment is flat / unusable
    """
    lo, hi = seg_bif.min(), seg_bif.max()
    if hi - lo < 1e-6:
        return None   # flat segment — carries no shape information

    # 1. Normalise to [0, 1]
    normed = (seg_bif - lo) / (hi - lo)

    # 2. Sample N_SAMPLE_PTS uniformly
    x = np.linspace(0, 1, N_SAMPLE_PTS)
    y = np.interp(x, np.linspace(0, 1, len(normed)), normed)

    # 3. Fit degree 1, 2, 3 and pick best by penalised error
    best_score  = np.inf
    best_coeffs = None

    for deg in [1, 2, 3]:
        coeffs = np.polyfit(x, y, deg)
        y_hat  = np.polyval(coeffs, x)
        mse    = np.mean((y - y_hat) ** 2)
        score  = mse + lambda_penalty * deg * np.sum(np.abs(coeffs))

        if score < best_score:
            best_score  = score
            best_coeffs = coeffs

    # 4. Always return exactly 3 values [a, b, c]
    #    degree-1 → 2 coeffs, degree-2 → 3 coeffs, degree-3 → 4 coeffs
    #    We keep the first 3 significant coefficients (drop intercept d
    #    for degree-3 since it is just the normalisation offset).
    padded = np.zeros(3)
    kept   = best_coeffs[:3]           # up to 3 highest-degree coefficients
    padded[-len(kept):] = kept[::-1]   # fill right-to-left, lowest degree first
    padded = padded[::-1]              # return [a(cubic), b(quad), c(linear)]
    return padded


def extract_features(segments):
    """
    Extract 3-D polynomial features from all segments of a single trace.

    Parameters
    ----------
    segments : list of (t_seg, bif_seg) tuples from segment_bif()

    Returns
    -------
    numpy array of shape (n_valid_segments, 3)
    Empty array of shape (0, 3) if no valid segments.
    """
    features = []
    for _, seg_bif in segments:
        coeffs = fit_segment(seg_bif)
        if coeffs is not None:
            features.append(coeffs)

    return np.array(features) if features else np.empty((0, 3))


def _get_features_from_csv(csv_path, rtt_s, server_ip='10.0.0.1'):
    """
    Internal helper: load CSV → compute BiF → smooth → remove slow start
    → segment → extract features. Returns array of shape (n_segs, 3).
    """
    t, bif         = compute_bif(csv_path, server_ip)
    t_s, bif_s     = smooth_bif(t, bif, rtt_s)
    t_ss, bif_ss   = remove_slow_start(t_s, bif_s)
    segments       = segment_bif(t_ss, bif_ss)
    return extract_features(segments)   # (n_segs, 3)


def extract_features_dual_profile(csv_50ms, csv_100ms,
                                   server_ip='10.0.0.1'):
    """
    Extract a 6-D feature vector by concatenating features from BOTH
    the 50ms and 100ms delay profiles.

    Why two profiles?  (paper §3.3)
    Some CCAs produce similar BiF shapes at one delay but differ at another.
    For example:
      - New Reno, Illinois, HSTCP look identical at 50ms delay
      - At 100ms their oscillation periods and amplitudes diverge
    Using both profiles doubles the discriminating information available
    to GNB.

    Parameters
    ----------
    csv_50ms  : path to CSV captured with 50ms one-way delay
    csv_100ms : path to CSV captured with 100ms one-way delay
    server_ip : IP of the TCP sender

    Returns
    -------
    feats_6d : numpy array of shape (n_paired_segments, 6)
               [a1,b1,c1,  a2,b2,c2]
               where 1 = 50ms profile, 2 = 100ms profile
    n_paired : int — number of matched segment pairs
    """
    # RTT = 2 × one-way delay
    f50  = _get_features_from_csv(csv_50ms,  rtt_s=0.10, server_ip=server_ip)
    f100 = _get_features_from_csv(csv_100ms, rtt_s=0.20, server_ip=server_ip)

    n = min(len(f50), len(f100))
    if n == 0:
        return np.empty((0, 6)), 0

    # Pair segments positionally (both traces are from the same CCA
    # under the same conditions — segments align by oscillation index)
    feats_6d = np.hstack([f50[:n], f100[:n]])   # (n, 6)
    return feats_6d, n