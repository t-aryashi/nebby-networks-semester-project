#!/bin/bash
# quic_dataset.sh — Self-contained QUIC dataset generator
#
# USAGE:
#   sudo caddy start --config /etc/caddy/Caddyfile
#   bash quic_dataset.sh
#
# WHAT WORKS (from debugging):
#   - All paths in /tmp (no spaces in path)
#   - Script files passed to mm-link, NOT bash -c (mm-link treats -c as its own flag)
#   - mm-link trace trace bash /tmp/script.sh  ← correct
#   - Python trace generation (bash integer math drops packets)
#   - tshark uses || not "in {}" (older tshark doesn't support set syntax)
#   - Caddy restarted before each run (it drops after first connection)

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
CCS=("reno" "cubic" "bbr")
DELAYS=(50 100)
BW=2000
BUFF_MUL=20
AQM="droptail"
QUIC_URL="https://10.0.0.1/10MB.zip"
CLIENT_TIMEOUT=120
CADDY_CONF="/etc/caddy/Caddyfile"

# All temp files in /tmp — no spaces, accessible inside Mahimahi
TMP="/tmp/quic_dataset"
TRACEFILE="$TMP/bw.trace"
INNER_SH="$TMP/inner.sh"
BTL_SH="$TMP/btl.sh"

# Output dir — resolved once upfront
OUTDIR="$(realpath -m "$(dirname "$0")/../candidates-measurements-quic")"

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

find_quiche_client() {
    if command -v quiche-client &>/dev/null; then
        command -v quiche-client; return 0
    fi
    for p in \
        "../../quiche/target/release/quiche-client" \
        "../../../quiche/target/release/quiche-client" \
        "$HOME/quiche/target/release/quiche-client" \
        "/opt/quiche/target/release/quiche-client"
    do
        [[ -f "$p" && -x "$p" ]] && { realpath "$p"; return 0; }
    done
    return 1
}

make_trace() {
    # Python avoids bash integer rounding — bash drops packets at short intervals
    # e.g. 2Mbps = 6ms interval, bash produces ~5500 lines instead of 20000
    python3 - << PYEOF
bw_kbps = $BW
interval_ms = (1500 * 8) / (bw_kbps * 1000) * 1000
duration_ms = 120 * 1000
t = 0.0
with open("$TRACEFILE", "w") as f:
    while t < duration_ms:
        f.write(str(int(t)) + "\n")
        t += interval_ms
PYEOF
    local lines; lines=$(wc -l < "$TRACEFILE")
    echo "  Trace: $TRACEFILE ($lines packets, 120s)"
}

restart_caddy() {
    # Caddy drops UDP after each QUIC connection in some versions.
    # Kill and restart cleanly before every run.
    sudo pkill caddy 2>/dev/null || true
    sleep 1
    sudo caddy start --config "$CADDY_CONF" > /dev/null 2>&1
    sleep 2
    if sudo ss -ulnp 2>/dev/null | grep -q ':443'; then
        echo "  ✓ Caddy up on UDP 443"
    else
        echo "  ✗ Caddy failed to start — aborting run"
        return 1
    fi
}

pcap_to_csv() {
    local pcap=$1 csv=$2
    # Use || syntax — older tshark doesn't support "udp.port in {443 80 ...}"
    tshark \
        -r "$pcap" \
        -Y "udp.port==443 || udp.port==4433 || udp.port==8443" \
        -T fields \
        -e frame.time_relative \
        -e ip.src \
        -e ip.dst \
        -e udp.length \
        -E header=y \
        -E separator=, \
        -E quote=d \
        -E occurrence=f \
        > "$csv"
    local lines; lines=$(wc -l < "$csv")
    echo "  CSV rows (incl. header): $lines"
    [[ $lines -ge 2 ]]
}

cleanup() { rm -f "$INNER_SH" "$BTL_SH"; }
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════
#  ONE MEASUREMENT RUN
# ══════════════════════════════════════════════════════════════════════

run_one() {
    local cc=$1 delay=$2 quiche_bin=$3

    local rtt=$(( delay * 2 ))
    local bw_bytes=$(( BW * 1000 / 8 ))
    local bdp=$(( bw_bytes * rtt / 1000 ))
    local buff=$(( (bdp * BUFF_MUL) / 10 ))
    [[ $buff -lt 1500 ]] && buff=1500
    echo "  RTT=${rtt}ms  BDP=${bdp}B  buffer=${buff}B"

    local pcap="$TMP/test_${cc}_${delay}.pcap"
    rm -f "$pcap"

    # Restart Caddy before every run — it drops after first connection
    restart_caddy || return 1

    # ── inner.sh — quiche-client (runs in innermost Mahimahi shell) ──
    # No bash -c, no spaces in any path, no nested quotes needed
    cat > "$INNER_SH" << EOF
#!/bin/bash
set -euo pipefail
timeout $CLIENT_TIMEOUT $quiche_bin \\
    --no-verify \\
    --wire-version 00000001 \\
    $QUIC_URL > /dev/null
EXIT=\$?
if   [[ \$EXIT -eq 124 ]]; then echo "  ERROR: timed out after ${CLIENT_TIMEOUT}s"; exit 1
elif [[ \$EXIT -ne 0   ]]; then echo "  ERROR: quiche-client exited \$EXIT"; exit \$EXIT
fi
echo "  Download complete."
EOF
    chmod +x "$INNER_SH"

    # ── btl.sh — tcpdump + mm-link + inner mm-delay ──────────────────
    # IMPORTANT: mm-link command must be a plain script path.
    # mm-link treats -c as its own flag so "bash -c '...'" breaks it.
    cat > "$BTL_SH" << EOF
#!/bin/bash
set -euo pipefail

tcpdump -i ingress -s 96 -w $pcap -q udp &
DUMP_PID=\$!
sleep 0.3

mm-link $TRACEFILE $TRACEFILE \\
    --uplink-queue=$AQM   --uplink-queue-args=bytes=$buff \\
    --downlink-queue=$AQM --downlink-queue-args=bytes=$buff \\
    mm-delay $delay \\
    bash $INNER_SH

sleep 1
kill "\$DUMP_PID" 2>/dev/null || true
wait "\$DUMP_PID" 2>/dev/null || true

if [[ -f $pcap && -s $pcap ]]; then
    echo "  pcap: \$(du -h $pcap | cut -f1)"
else
    echo "  WARNING: pcap missing or empty"
fi
EOF
    chmod +x "$BTL_SH"

    # ── Launch: outer mm-delay → btl.sh ──────────────────────────────
    # This is the exact pattern confirmed working in debugging:
    #   mm-delay 50 bash _btl.sh
    # where _btl.sh contains: mm-link ... bash _inner.sh
    mm-delay "$delay" bash "$BTL_SH"

    # ── pcap → CSV ───────────────────────────────────────────────────
    if [[ -f "$pcap" && -s "$pcap" ]]; then
        local tmp_csv="$TMP/out_${cc}_${delay}.csv"
        echo "  Converting pcap → CSV ..."
        if pcap_to_csv "$pcap" "$tmp_csv"; then
            mkdir -p "$OUTDIR"
            local ts; ts=$(date +%s)
            local out="$OUTDIR/cc-${cc}_aqm-${AQM}_bw-${BW}_buf-${BUFF_MUL}_${ts}_quic.csv"
            mv "$tmp_csv" "$out"
            echo "  ✓ Saved: $out"
            rm -f "$pcap"
            return 0
        else
            echo "  ✗ CSV empty — no QUIC traffic captured"
            rm -f "$pcap" "$tmp_csv"
            return 1
        fi
    else
        echo "  ✗ No pcap produced"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

mkdir -p "$TMP"

echo "=================================================="
echo "  QUIC Dataset Generator"
echo "  CCAs   : ${CCS[*]}"
echo "  Delays : ${DELAYS[*]} ms"
echo "  BW     : ${BW} Kbps   BufMul: ${BUFF_MUL}   AQM: ${AQM}"
echo "  URL    : ${QUIC_URL}"
echo "  Output : $OUTDIR"
echo "=================================================="
echo ""

QUICHE_BIN=$(find_quiche_client) || {
    echo "✗ quiche-client not found. Run: bash install_quiche.sh ../../quiche"
    exit 1
}
echo "✓ quiche-client : $QUICHE_BIN"
echo ""

echo "Generating trace ..."
make_trace
echo ""

mkdir -p "$OUTDIR"
PASS=0; FAIL=0

for cc in "${CCS[@]}"; do
    for delay in "${DELAYS[@]}"; do
        echo "──────────────────────────────────────────────"
        echo "  CC=${cc}  Delay=${delay}ms"
        if run_one "$cc" "$delay" "$QUICHE_BIN"; then
            (( PASS++ )) || true
        else
            (( FAIL++ )) || true
        fi
        echo ""
    done
done

echo "=================================================="
echo "  Done.  ✓ $PASS passed   ✗ $FAIL failed"
echo "  Output: $OUTDIR"
ls "$OUTDIR" 2>/dev/null | tail -10 || true
echo "=================================================="