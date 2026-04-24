"""
predict.py
-----------
Classify a single tshark CSV file using trained Nebby models.

Usage:
    python predict.py --csv path/to/trace_tcp.csv \
                      --model_dir ../models

Outputs the predicted CC and AQM, plus a BiF plot.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Import preprocessing functions
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    compute_bif, estimate_rtt, fft_smooth,
    extract_segments, normalize_and_sample, fit_polynomial,
    extract_periodicity_features, parse_label
)

ALL_FEATURES = [
    'coeff_a', 'coeff_b', 'coeff_c', 'coeff_d',
    'seg_mean', 'seg_std', 'seg_amp',
    'bw_norm', 'buf_norm', 'rtt_norm',
    'probe_period_s', 'drain_period_s', 'spike_count', 'bif_cv',
    'aqm_codel',
]


def load_model(model_dir: str, target: str):
    path = os.path.join(model_dir, f'model_{target}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def classify_trace(csv_path: str, model_dir: str, plot: bool = True):
    """
    Full inference pipeline for one CSV file.
    """
    print(f"\nClassifying: {csv_path}")

    # Try to parse ground-truth label from filename (may fail for new files)
    try:
        meta = parse_label(os.path.basename(csv_path))
        gt_cc  = meta['cc']
        gt_aqm = meta['aqm']
        bw_norm  = meta['bw'] / 10000.0
        buf_norm = meta['buf_mul'] / 50.0
    except ValueError:
        meta     = None
        gt_cc    = 'unknown'
        gt_aqm   = 'unknown'
        bw_norm  = 0.5
        buf_norm = 0.2

    # ── Read and process ──────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if len(df) < 50:
        print("  Too few packets to classify.")
        return

    bif_raw = compute_bif(df)
    rtt_s   = estimate_rtt(df)
    rtt_s   = max(0.005, min(rtt_s, 2.0))

    bif_ca  = bif_raw[bif_raw.index > 3.0]
    if len(bif_ca) < 100:
        bif_ca = bif_raw

    bif_smooth    = fft_smooth(bif_ca, rtt_s=rtt_s)
    period_feats  = extract_periodicity_features(bif_smooth.values)
    segments      = extract_segments(bif_smooth)
    if not segments:
        segments = [bif_smooth.values]

    delay_norm = rtt_s / 0.12
    aqm_codel  = 0  # unknown at inference time; use 0

    # Build feature rows
    rows = []
    for seg in segments:
        normed = normalize_and_sample(seg)
        coeffs = fit_polynomial(normed)
        row = {
            'coeff_a': coeffs[0], 'coeff_b': coeffs[1],
            'coeff_c': coeffs[2], 'coeff_d': coeffs[3],
            'seg_mean': np.mean(normed), 'seg_std': np.std(normed),
            'seg_amp':  np.max(normed) - np.min(normed),
            'bw_norm':  bw_norm, 'buf_norm': buf_norm,
            'rtt_norm': delay_norm,
            'probe_period_s': period_feats['probe_period_s'],
            'drain_period_s': period_feats['drain_period_s'],
            'spike_count':    period_feats['spike_count'],
            'bif_cv':         period_feats['bif_cv'],
            'aqm_codel':      aqm_codel,
        }
        rows.append(row)

    X = pd.DataFrame(rows)

    # ── Load models and predict ───────────────────────────────────────────
    predictions = {}
    for target in ['cc', 'aqm']:
        payload = load_model(model_dir, target)
        if payload is None:
            print(f"  No model found for {target}. Run train.py first.")
            continue

        feat_cols = payload['feature_names']
        feat_cols = [c for c in feat_cols if c in X.columns]
        X_feat    = X[feat_cols].fillna(0)

        pipeline = payload['pipeline']
        le       = payload['label_encoder']

        # Per-segment predictions, then majority vote
        y_pred_enc  = pipeline.predict(X_feat.values)
        y_pred_str  = le.inverse_transform(y_pred_enc)

        # Probability-weighted vote
        y_proba = pipeline.predict_proba(X_feat.values)
        mean_proba = y_proba.mean(axis=0)
        best_idx   = np.argmax(mean_proba)
        best_class = le.inverse_transform([best_idx])[0]
        confidence = mean_proba[best_idx]

        # Unanimous check (paper's rule: if segments disagree → Unknown)
        unique_preds = np.unique(y_pred_str)
        if len(unique_preds) > 1:
            final = f"Unknown (disagreement: {unique_preds})"
        else:
            final = best_class

        predictions[target] = {
            'prediction': final,
            'confidence': confidence,
            'per_segment': y_pred_str.tolist(),
        }

    # ── Print result ──────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print(f"  Estimated RTT:      {rtt_s*1000:.1f} ms")
    print(f"  Segments found:     {len(segments)}")
    for target, info in predictions.items():
        gt = gt_cc if target == 'cc' else gt_aqm
        correct = '✓' if info['prediction'] == gt else '✗'
        print(f"  {target.upper():5s} prediction:  {info['prediction']:15s} "
              f"(conf={info['confidence']:.2f})  "
              f"GT={gt} {correct}")
    print("─" * 45)

    # ── Optional BiF plot ─────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                 gridspec_kw={'height_ratios': [2, 1]})

        ax1 = axes[0]
        ax1.plot(bif_raw.index, bif_raw.values / 1024,
                 color='lightgray', linewidth=0.5, label='Raw BiF')
        ax1.plot(bif_smooth.index, bif_smooth.values / 1024,
                 color='steelblue', linewidth=1.2, label='Smoothed BiF')

        # Mark segments
        colors = plt.cm.Set1(np.linspace(0, 0.8, len(segments)))
        t_cursor = bif_smooth.index[0] + 3.0
        for i, seg in enumerate(segments):
            seg_t = np.linspace(t_cursor,
                                t_cursor + len(seg) / 500,
                                len(seg))
            ax1.axvspan(seg_t[0], seg_t[-1], alpha=0.1,
                        color=colors[i], label=f'Seg {i+1}' if i < 3 else '')
            t_cursor = seg_t[-1]

        cc_pred  = predictions.get('cc',  {}).get('prediction', '?')
        aqm_pred = predictions.get('aqm', {}).get('prediction', '?')
        ax1.set_title(f"BiF Trace — Predicted CC: {cc_pred}  |  "
                      f"AQM: {aqm_pred}  |  "
                      f"Ground Truth CC: {gt_cc}", fontsize=11)
        ax1.set_ylabel('KBytes in Flight')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Gradient plot (used for back-off detection)
        ax2 = axes[1]
        grad = np.gradient(bif_smooth.values / 1024)
        ax2.plot(bif_smooth.index, grad, color='tomato',
                 linewidth=0.7, alpha=0.8)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('d(BiF)/dt')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('BiF Gradient (negative spikes = back-off events)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = csv_path.replace('.csv', '_bif.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  BiF plot saved: {out}")

    return predictions


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify a tshark CSV file with trained Nebby models')
    parser.add_argument('--csv',       required=True,
                        help='Path to *_tcp.csv file from pcap2csv.sh')
    parser.add_argument('--model_dir', default='../models',
                        help='Folder with model_cc.pkl / model_aqm.pkl')
    parser.add_argument('--no_plot',   action='store_true',
                        help='Skip BiF plot generation')
    args = parser.parse_args()

    classify_trace(args.csv, args.model_dir, plot=not args.no_plot)