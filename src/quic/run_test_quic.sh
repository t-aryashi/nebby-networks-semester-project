#!/bin/bash
# run_test_quic.sh
#
# Mirror of run_test.sh — orchestrates one QUIC measurement run.
#
# DIFFERENCES FROM run_test.sh
# ──────────────────────────────
#  1. Calls simnet_quic.sh instead of simnet.sh
#     (which calls btl_quic.sh → client_quic.sh).
#  2. Converts pcap with pcap2csv_quic.sh (extracts udp.length, not tcp.seq).
#  3. Output CSV is named *_quic.csv and saved to ../candidates-measurements-quic/
#
# ARGUMENTS (same positions as run_test.sh)
#   $1  cc          — CCA name (passed to quiche-client --cc)
#   $2  predelay    — outer mm-delay (ms)
#   $3  postdelay   — inner mm-delay (ms)
#   $4  linkspeed   — bottleneck link speed (Kbps)
#   $5  buff_mul    — buffer size multiplier (buff = BDP × buff_mul/10)
#   $6  url         — QUIC target URL (https://...)
#   $7  aqm         — AQM type (droptail / pfifo)

set -euo pipefail

cc=$1
predelay=$2
postdelay=$3
linkspeed=$4
buff_mul=$5
url=$6
aqm=$7

# ── Cleanup from previous run ─────────────────────────────────────────────────
rm -f test_quic.pcap test_quic.pcap-udp.csv

# ── Run the QUIC simulation ───────────────────────────────────────────────────
# In run_test_quic.sh, change the simnet call to have a timeout
timeout 120 bash simnet_quic.sh "$cc" "$predelay" "$postdelay" "$linkspeed" "$buff_mul" "$url" "$aqm" || {
    echo "  WARNING: run timed out or failed for cc=$cc delay=$predelay"
}

# ── Convert pcap → CSV ────────────────────────────────────────────────────────
if [ -f test_quic.pcap ] && [ -s test_quic.pcap ]; then
    bash pcap2csv_quic.sh test_quic.pcap

    outdir="../candidates-measurements-quic"
    mkdir -p "$outdir"

    ts=$(date +%s)
    # Filename IS the label for train_quic.py — same pattern as TCP
    label="cc-${cc}_aqm-${aqm}_bw-${linkspeed}_buf-${buff_mul}"
    output="${outdir}/${label}_${ts}_quic.csv"

    cp test_quic.pcap-udp.csv "$output"
    echo "  Saved: ${output}"
else
    echo "  ERROR: Simulation did not produce test_quic.pcap — skipping CSV step."
    echo "  Likely causes:"
    echo "    1. quiche-client could not connect (is the QUIC server running?)"
    echo "    2. URL is unreachable from inside Mahimahi"
    echo "    3. TLS certificate rejected (pass --no-verify in client_quic.sh)"
    exit 1
fi