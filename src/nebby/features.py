"""
features.py — Polynomial feature extraction from BiF segments
Paper reference: Nebby §3.4 steps 3 & 4 (Sampling, Polynomial Fitting,
                 Clustering and Classification)

Each segment is:
  1. Normalised to [0, 1]
  2. Resampled to 200 uniformly-spaced points
  3. Fit with degree-1, 2, and 3 polynomials
  4. Best fit chosen by penalised MSE  (Lasso-like, λ = 0.7)
  5. Coefficients [a, b, c] used as the 3-D feature vector

Table 2 from the paper:
  Linear  : BIC, YeAH, Scalable, HSTCP, Vegas, Veno
  Quadratic: Illinois, New Reno, Westwood
  Cubic   : CUBIC, HTCP
"""

import numpy as np


def fit_segment(seg_bif, lambda_penalty=0.7):
    """
    Fit a normalised BiF segment with polynomials of degree 1 / 2 / 3.
    Pick the best by penalised MSE and return 3 coefficients [a, b, c].

    Error score (paper eq.):
        Error = MSE + λ × Degree × Sum(|coefficients|)

    Parameters
    ----------
    seg_bif        : 1-D array of BiF values for one segment
    lambda_penalty : regularisation weight (paper uses 0.7)

    Returns
    -------
    numpy array of shape (3,)  →  [a, b, c]
    or None if the segment is flat / unusable
    """
    lo, hi = seg_bif.min(), seg_bif.max()
    if hi - lo < 1e-6:
        return None   # constant segment — skip

    # 1. Normalise to [0, 1]
    normed = (seg_bif - lo) / (hi - lo)

    # 2. Sample 200 uniformly-spaced points
    x = np.linspace(0, 1, 200)
    y = np.interp(x, np.linspace(0, 1, len(normed)), normed)

    # 3 & 4. Fit and score each degree
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

    # 5. Pad to always return exactly 3 values [a, b, c]
    #    degree-1 poly has 2 coeffs → pad with leading zero
    #    degree-2 poly has 3 coeffs → already fine
    #    degree-3 poly has 4 coeffs → keep first 3 (a, b, c; drop intercept d)
    padded = np.zeros(3)
    kept   = best_coeffs[:3]          # drop intercept for deg-3
    padded[-len(kept):] = kept[::-1]  # fill from the right, lowest degree first
    padded = padded[::-1]             # return as [a (cubic), b (quad), c (linear)]
    return padded


def extract_features(segments):
    """
    Extract polynomial features from all segments of a single trace.

    Parameters
    ----------
    segments : list of (t_seg, bif_seg) tuples from segment_bif()

    Returns
    -------
    numpy array of shape (n_valid_segments, 3)
    Empty array of shape (0, 3) if no valid segments found.
    """
    features = []
    for _, seg_bif in segments:
        coeffs = fit_segment(seg_bif)
        if coeffs is not None:
            features.append(coeffs)

    return np.array(features) if features else np.empty((0, 3))