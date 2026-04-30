#!/bin/bash
# install_quiche.sh
#
# Symlinks the quiche-client binary to /usr/local/bin/quiche-client.
#
# BINARY LOCATION
# ─────────────────────────────────────────────────────────────────────
# The quiche workspace uses a SHARED target/ directory one level above apps/.
# After running "cd quiche/apps && cargo build --release", the binary lands at:
#
#   quiche/target/release/quiche-client        ← correct (shared workspace target)
#   quiche/apps/target/release/quiche-client   ← does NOT exist
#
# From your src/quic/ directory, the path is:
#   ../../quiche/target/release/quiche-client
#
# USAGE
#   bash install_quiche.sh [path_to_quiche_repo]
#   bash install_quiche.sh ../../quiche          ← your case

set -euo pipefail

INSTALL_PATH="/usr/local/bin/quiche-client"
QUICHE_REPO="${1:-}"

echo "=================================================="
echo "  quiche-client installation"
echo "=================================================="
echo ""

# ── Find quiche repo ──────────────────────────────────────────────────────────
find_repo() {
    local candidates=(
        "$QUICHE_REPO"
        "../../quiche"
        "../quiche"
        "./quiche"
        "$HOME/quiche"
        "/opt/quiche"
    )
    for d in "${candidates[@]}"; do
        [[ -z "$d" ]] && continue
        [[ -f "$d/Cargo.toml" ]] && { realpath "$d"; return 0; }
    done
    return 1
}

REPO=$(find_repo) || {
    echo "ERROR: Could not find quiche repo."
    echo "  Usage: bash install_quiche.sh /path/to/quiche"
    exit 1
}
echo "  Quiche repo : $REPO"

# ── The binary is in the WORKSPACE target/, not apps/target/ ─────────────────
# quiche/apps/ shares the parent workspace target directory.
# So "cd apps && cargo build --release" puts binaries in ../target/release/
FULL_BIN="$REPO/target/release/quiche-client"

if [[ ! -f "$FULL_BIN" ]]; then
    echo ""
    echo "  Binary not found at: $FULL_BIN"
    echo "  Building now (this takes ~2 minutes) ..."
    echo ""
    cd "$REPO/apps"
    cargo build --release
    cd - > /dev/null
fi

if [[ ! -f "$FULL_BIN" ]]; then
    echo "ERROR: Build completed but binary still not found."
    echo "  Expected: $FULL_BIN"
    echo "  Contents of $REPO/target/release/:"
    ls "$REPO/target/release/" | grep -v "^lib\|\.d$" | head -20 || true
    exit 1
fi

echo "  Found binary: $FULL_BIN"
echo ""

# ── Verify it's the full app ──────────────────────────────────────────────────
HELP=$("$FULL_BIN" --help 2>&1 || true)
if echo "$HELP" | grep -q "no-verify\|no_verify"; then
    echo "  ✓ --no-verify flag confirmed"
else
    echo "  Help output:"
    echo "$HELP" | head -20
    echo ""
    echo "  ✗ --no-verify not found. Binary may be wrong."
    exit 1
fi

if echo "$HELP" | grep -q "\-\-cc\b"; then
    echo "  ✓ --cc flag confirmed"
else
    echo "  ⚠ --cc not found — CCA selection unavailable"
fi

# ── Symlink ───────────────────────────────────────────────────────────────────
echo ""
echo "  Symlinking → $INSTALL_PATH"
sudo ln -sf "$FULL_BIN" "$INSTALL_PATH"

echo ""
echo "  ✓ Done. Test with:"
echo "    quiche-client --no-verify --cc cubic --wire-version 1 \\"
echo "                  --dump-responses /dev/null https://10.0.0.1/10MB.zip"
echo ""
echo "  Then run:"
echo "    bash generate_dataset_quic.sh"
echo "=================================================="