#!/bin/bash
# generate_dataset_quic.sh
#
# Mirror of generate_dataset.sh — QUIC/HTTP3 version.
#
# PRE-REQUISITES (one-time, run before this script)
#   sudo bash setup_quic.sh          ← start Caddy HTTP/3 server
#   bash install_quiche.sh ../../quiche  ← install quiche-client

set -euo pipefail

CCS=("reno" "cubic" "bbr" "bic" "htcp" "hybla" "illinois" "scalable" "vegas" "veno" "westwood" "yeah")
DELAYS=(50 100)
BW=2000
BUFF_MUL=20
AQM="droptail"
QUIC_URL="https://10.0.0.1/10MB.zip"

mkdir -p ../candidates-measurements-quic
mkdir -p ../traces

echo "=================================================="
echo "  Nebby QUIC Dataset Generation"
echo "  CCAs    : ${CCS[*]}"
echo "  Delays  : ${DELAYS[*]} ms"
echo "  BW      : ${BW} Kbps  BufMul: ${BUFF_MUL} (=2×BDP)  AQM: ${AQM}"
echo "  URL     : ${QUIC_URL}"
echo "=================================================="
echo ""

# # ── Server sanity check ───────────────────────────────────────────────────────
# # Use exit code only — --dump-responses suppresses all stdout so grep is useless.
# # Wire version must be hex: 00000001 for QUIC v1 (NOT the integer 1).
# echo "Checking QUIC server is reachable ..."
# if command -v quiche-client &>/dev/null; then
#     if quiche-client \
#             --no-verify \
#             --wire-version 00000001 \
#             "$QUIC_URL" 2>/dev/null; then
#         echo "  ✓ QUIC server is up"
#     else
#         echo "  ✗ QUIC server not responding (exit code $?)."
#         echo ""
#         echo "  Debug — try manually (shows full error output):"
#         echo "    quiche-client --no-verify --wire-version 00000001 $QUIC_URL"
#         echo ""
#         echo "  Common fixes:"
#         echo "    1. Server not running:  sudo bash setup_quic.sh"
#         echo "    2. Caddy not on port 443: ss -tlnp | grep 443"
#         echo "    3. Wrong wire version:  try --wire-version ff00001d  (draft-29)"
#         exit 1
#     fi
# else
#     echo "  ⚠ quiche-client not on PATH — skipping check."
#     echo "    Run: bash install_quiche.sh ../../quiche"
# fi
# echo ""

# ── Main measurement loop ─────────────────────────────────────────────────────
for cc in "${CCS[@]}"; do
    for delay in "${DELAYS[@]}"; do
        echo "------------------------------------------------"
        echo "GENERATE QUIC: CC=$cc  Delay=${delay}ms  BW=${BW}  Buf=2xBDP"

        sudo sysctl -w net.ipv4.tcp_congestion_control="$cc" > /dev/null 2>&1 || true

        bash run_test_quic.sh "$cc" "$delay" "$delay" "$BW" "$BUFF_MUL" "$QUIC_URL" "$AQM"
    done
done

echo ""
echo "=================================================="
echo "  QUIC dataset generation complete."
echo "  CSVs saved to: ../candidates-measurements-quic/"
ls ../candidates-measurements-quic/ | tail -10
echo "=================================================="