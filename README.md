# Nebby — CCA Identification Tool (SIGCOMM '24 Reimplementation)

Reimplementation of **"Keeping an Eye on Congestion Control in the Wild with Nebby"**  
Mishra et al., ACM SIGCOMM 2024.

---

## Project Structure

```
project/
├── src/
│   └── datasetgeneration/
│       ├── generate_dataset.sh   # outer loop: CCA × delay
│       ├── run_test.sh           # per-test orchestrator
│       ├── simnet.sh             # BDP/buffer calc + trace gen + mm-delay
│       ├── btl.sh                # bottleneck: mm-link + tcpdump
│       ├── client.sh             # wget inside Mahimahi
│       ├── pcap2csv.sh           # tshark pcap → CSV
│       └── start_server.sh       # Python HTTP server (10MB file)
│
├── nebby/
│   ├── bif.py                    # Step 1 — compute & smooth BiF
│   ├── preprocess.py             # Step 2 — remove slow start, segment
│   ├── features.py               # Step 3 — polynomial feature extraction
│   ├── train.py                  # Step 4 — build dataset, train GNB
│   ├── classify.py               # Step 5 — classify a single trace
│   └── evaluate.py               # Step 6 — confusion matrix + plots
│
├── candidates-measurements/      # CSVs produced by generate_dataset.sh
├── models/                       # gnb.pkl + label_encoder.pkl (after training)
├── evaluation/                   # plots produced by evaluate.py
└── traces/                       # Mahimahi bandwidth trace files
```

---

## Quick Start

### Prerequisites

```bash
# System
sudo apt install mahimahi tcpdump tshark python3-pip

# Python
pip install numpy pandas scipy scikit-learn matplotlib joblib
```

### 1 — Start the server

```bash
cd src/datasetgeneration
sudo bash start_server.sh
```

This creates `10MB.zip` and serves it on port 80.  
Inside Mahimahi the server is reachable at `http://10.0.0.1/10MB.zip`
(or `http://100.64.0.1/10MB.zip` on bare-metal Linux — check your env).

### 2 — Generate dataset

```bash
cd src/datasetgeneration
./generate_dataset.sh
```

This runs every combination of `(CCA, delay)` and saves CSVs to
`../candidates-measurements/`.

**Run this multiple times** to accumulate enough segments for training.  
The paper used 50 runs per CCA per vantage point. Aim for at least 10 runs.

### 3 — Train the classifier

```bash
cd nebby
python3 train.py
```

Outputs `../models/gnb.pkl` and `../models/label_encoder.pkl`.

### 4 — Evaluate

```bash
python3 evaluate.py
```

Saves three plots to `../evaluation/`:
- `confusion_matrix.png`
- `bif_traces_eval.png`
- `confidence_histogram.png`

### 5 — Classify a new trace

```bash
python3 classify.py ../candidates-measurements/cc-cubic_aqm-droptail_bw-200_buf-20_123_tcp.csv
```

---

## How It Works (Paper §3)

```
Raw pcap
  │
  ▼
bif.py: compute_bif()
  │  BiF(t) = cummax(seq + len) − cummax(ack)
  │  separated by direction (server IP vs client IP)
  │
  ▼
bif.py: smooth_bif()
  │  Low-pass Butterworth filter at cutoff = 1/RTT
  │  Removes ACK compression noise (sub-RTT variation)
  │
  ▼
preprocess.py: remove_slow_start()
  │  Detects first >40% BiF drop = first loss event
  │  Discards everything before it
  │
  ▼
preprocess.py: segment_bif()
  │  Splits at every >35% BiF drop (back-off events)
  │  Each segment = one oscillation cycle
  │
  ▼
features.py: fit_segment()
  │  Normalise segment to [0,1]
  │  Sample 200 points
  │  Fit poly degree 1/2/3
  │  Score = MSE + 0.7 × degree × Σ|coefficients|
  │  Return best [a, b, c]
  │
  ▼
classify.py / train.py
     BBR?  →  rule-based: check ProbeBW (every 8 RTTs) + ProbeRTT period
     else  →  Gaussian Naive Bayes on [a, b, c] coefficients
```

---

## Key Parameters

| Parameter | Where | Value | Why |
|---|---|---|---|
| Bottleneck BW | `generate_dataset.sh` | 200 Kbps | Paper §3.3 |
| Buffer | `simnet.sh` | 2 × BDP | Paper §3.3 |
| One-way delays | `generate_dataset.sh` | 50 ms, 100 ms | Paper §3.3 (two profiles) |
| AQM | `generate_dataset.sh` | droptail | Paper §3.3 |
| RTT for smoothing | `bif.py` | 0.1 s | 2 × 50 ms one-way |
| λ penalty | `features.py` | 0.7 | Paper §3.4 |
| Poly sample points | `features.py` | 200 | Paper §3.4 |
| Slow-start drop | `preprocess.py` | 40% | Heuristic |
| Segment drop | `preprocess.py` | 35% | Heuristic |
| Min segment length | `preprocess.py` | 1.0 s | Avoid tiny noise segments |

---

## Tuning Tips

**Not enough segments per class?**  
Run `generate_dataset.sh` more times. Each run adds new segments. The paper
used 250 measurements per CCA (50 runs × 5 vantage points).

**CUBIC and Reno look the same?**  
Lower `drop_fraction` in `segment_bif()` to catch more subtle back-offs,
or add more 100 ms delay traces (the second network profile helps separate them).

**BBR not detected?**  
Increase the RTT tolerance in `detect_bbr()`. At 200 Kbps the ProbeBW
interval may be longer than 8 × RTT due to buffering.

**Validation accuracy low?**  
Check `bif_traces_eval.png` — if the BiF shape looks flat (no sawtooth),
the capture point may still be before the bottleneck. See btl.sh fix in README.

---
<!-- 
## btl.sh — Critical Fix

The capture point **must be inside mm-link**, not before it.
If tcpdump runs before mm-link, it sees dropped packets and BiF is inflated.

```bash
# WRONG (capture before bottleneck):
tcpdump -i ingress -w "$dump" &
mm-link ... mm-delay ... ./client.sh

# CORRECT (capture after bottleneck, inside mm-link shell):
mm-link ../traces/bw.trace ../traces/bw.trace \
    --uplink-queue=$aqm --uplink-queue-args="bytes=$buff" \
    --downlink-queue=$aqm --downlink-queue-args="bytes=$buff" \
    bash -c "tcpdump -i ingress -w $dump -q & sleep 0.2; \
             mm-delay $postdelay ./client.sh $cc $link; \
             sleep 1; kill \$(jobs -p) 2>/dev/null"
``` -->

---

## References

- Mishra et al., *Keeping an Eye on Congestion Control in the Wild with Nebby*,
  ACM SIGCOMM 2024. https://doi.org/10.1145/3651890.3672223
- Official implementation: https://github.com/NUS-SNL/Nebby
- Mahimahi network emulator: http://mahimahi.mit.edu