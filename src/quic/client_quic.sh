#!/bin/bash
# client_quic.sh — runs INSIDE the innermost Mahimahi shell.
#
# BINARY PATH
# ─────────────────────────────────────────────────────────────────────
# quiche/apps/ uses the parent workspace target/:
#   quiche/target/release/quiche-client   ← correct
#   quiche/apps/target/release/...        ← does NOT exist
#
# ARGUMENTS
#   $1  cc    — CCA name
#   $2  link  — HTTPS URL

set -euo pipefail

cc=$1
link=$2

# ── Find binary ───────────────────────────────────────────────────────────────
find_client() {
    # 1. Installed on PATH by install_quiche.sh
    if command -v quiche-client &>/dev/null; then
        echo "quiche-client"; return 0
    fi
    # 2. Workspace target/ — this is where cargo puts it
    local candidates=(
        "../../quiche/target/release/quiche-client"
        "../../../quiche/target/release/quiche-client"
        "$HOME/quiche/target/release/quiche-client"
        "/opt/quiche/target/release/quiche-client"
    )
    for p in "${candidates[@]}"; do
        if [[ -f "$p" && -x "$p" ]]; then
            echo "$p"; return 0
        fi
    done
    return 1
}

QUICHE_BIN=$(find_client) || {
    echo "ERROR: quiche-client not found."
    echo "  Run: bash install_quiche.sh ../../quiche"
    echo "  (binary is at quiche/target/release/quiche-client)"
    exit 1
}

# ── CCA mapping ───────────────────────────────────────────────────────────────
case "$cc" in
    cubic|reno|bbr|bbr2) QUIC_CC="$cc" ;;
    bic|htcp|hybla|hstcp|scalable) QUIC_CC="cubic"
        echo "  NOTE: $cc → cubic" ;;
    illinois|westwood|veno|yeah|vegas) QUIC_CC="reno"
        echo "  NOTE: $cc → reno" ;;
    *) QUIC_CC="cubic"
        echo "  NOTE: unknown '$cc' → cubic" ;;
esac

# Inside Mahimahi, host is 100.64.0.1
# INNER_LINK=$(echo "$link" | sed 's|https://10\.0\.0\.1|https://100.64.0.1|g')
INNER_LINK="$link"

echo "  Binary : $QUICHE_BIN"
echo "  CC     : $QUIC_CC"
echo "  URL    : $INNER_LINK"

"$QUICHE_BIN" \
    --no-verify \
    --wire-version 00000001 \
    "$INNER_LINK"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "  ERROR: quiche-client exited $EXIT_CODE"
    echo "  Is the QUIC server running?  sudo bash setup_quic_server.sh"
    exit "$EXIT_CODE"
fi
echo "  Download complete."