"""
train_stream.py — Build dataset and train 3D GNB for Streaming Classification
Resolves the 6D vs 3D dimensionality crash in stream_classify.py
"""

import os, sys, glob, re
import numpy as np
import joblib
from collections import defaultdict
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))

from features import _get_features_from_csv

# ── config ────────────────────────────────────────────────────────────────────
CSV_DIR      = '../candidates-measurements'
MODEL_DIR    = '../models'
SERVER_IP    = '10.0.0.1'
MIN_SEGMENTS = 5      

RATE_BASED_CCAS = {'bbr', 'bbr2', 'bbr3'}

# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD SINGLE-PROFILE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_stream_dataset(csv_dir, server_ip=SERVER_IP):
    """
    Process every trace individually to build a 3D feature set.
    Returns X of shape (n_segments, 3) and y of shape (n_segments,).
    """
    files = sorted(glob.glob(os.path.join(csv_dir, '*_tcp.csv')))
    
    if not files:
        raise ValueError(f"No CSV files found in {csv_dir}.")

    X, y          = [], []
    per_class     = {}

    print(f"\n  Found {len(files)} trace file(s)\n")
    print(f"  {'CCA':<12} {'FILE':<50} {'SEGS':>5}")
    print(f"  {'─'*12} {'─'*50} {'─'*5}")

    for fpath in files:
        fname = os.path.basename(fpath)
        cc_match = re.search(r'cc-(\w+)_', fname)
        if not cc_match:
            continue
            
        label = cc_match.group(1)
        if label in RATE_BASED_CCAS:
            continue  # BBR handled via rule

        try:
            # For streaming training, we use a standard 100ms RTT smoothing baseline
            feats = _get_features_from_csv(fpath, rtt_s=0.10, server_ip=server_ip)
            n = len(feats)
        except Exception as e:
            print(f"  {label:<12} {fname:<50} ERROR: {e}")
            continue

        if n == 0:
            print(f"  {label:<12} {fname:<50} {'0':>5}")
            continue

        print(f"  {label:<12} {fname:<50} {n:>5}")
        per_class[label] = per_class.get(label, 0) + n

        for fv in feats:
            X.append(fv)
            y.append(label)

    return np.array(X), np.array(y), per_class

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  Nebby — Training Single-Profile (3D) Streaming Model")
    print("=" * 60)

    X, y, per_class = build_stream_dataset(CSV_DIR, SERVER_IP)

    if len(X) == 0:
        print("\nNo data collected.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Training Gaussian Naive Bayes (3D)")
    print("=" * 60)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    gnb = GaussianNB()
    gnb.fit(X, y_enc)

    joblib.dump(gnb, os.path.join(MODEL_DIR, 'gnb_stream.pkl'))
    joblib.dump(le,  os.path.join(MODEL_DIR, 'label_encoder_stream.pkl'))
    
    print(f"\n  Saved: {MODEL_DIR}/gnb_stream.pkl")
    print(f"  Saved: {MODEL_DIR}/label_encoder_stream.pkl")
    print(f"\n  GNB classes : {list(le.classes_)}")
    print(f"  Feature dim : {X.shape[1]}D  (single profile for streaming)\n")