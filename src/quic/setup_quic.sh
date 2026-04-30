#!/bin/bash
# setup_quic_server.sh
#
# ONE-TIME SETUP — run this on the host (10.0.0.1) BEFORE generate_dataset_quic.sh.
#
# WHAT THIS SCRIPT DOES
# ──────────────────────
# 1. Installs Caddy (a Go web server with native QUIC/HTTP3 support).
#    Caddy is the easiest way to get a working QUIC server:
#      - Built-in automatic TLS (self-signed cert for local use)
#      - HTTP/3 enabled by default
#      - No recompilation needed
#
# 2. Generates a 10 MB test file (same size as the TCP setup).
#
# 3. Writes a Caddyfile that serves the file over QUIC on port 443.
#
# 4. Starts caddy and prints a verification command.
#
# WHY CADDY INSTEAD OF NGINX?
# ────────────────────────────
# nginx requires a QUIC-enabled build (either nginx-quic branch or
# a custom patch) which is non-trivial to compile. Caddy ships with
# QUIC support in its standard binary — no patches needed.
#
# ALTERNATIVE: QUICHE'S OWN SERVER
# ──────────────────────────────────
# If you already have quiche built, you can run its example server:
#   cd quiche
#   cargo run --example http3-server -- \
#       --listen 0.0.0.0:4433 \
#       --root /var/www/quic/ \
#       --cert apps/src/bin/quiche-server.crt \
#       --key  apps/src/bin/quiche-server.key
# Then set QUIC_URL="https://10.0.0.1:4433/10MB.zip" in generate_dataset_quic.sh.
#
# USAGE
#   sudo bash setup_quic_server.sh

set -euo pipefail

WEB_ROOT="/var/www/quic"
CERT_DIR="/etc/caddy/certs"
CADDY_FILE="/etc/caddy/Caddyfile"
FILE_SIZE_MB=10

echo "=================================================="
echo "  Nebby QUIC Server Setup"
echo "=================================================="

# ── 1. Install Caddy ──────────────────────────────────────────────────────────
if ! command -v caddy &>/dev/null; then
    echo ""
    echo "Installing Caddy ..."
    # Official install from Caddy's apt repo
    apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -q
    apt-get install -y caddy
    echo "  Caddy installed: $(caddy version)"
else
    echo "  Caddy already installed: $(caddy version)"
fi

# ── 2. Generate self-signed TLS certificate ───────────────────────────────────
# QUIC requires TLS. We use a self-signed cert for the test server.
# quiche-client uses --no-verify so certificate validation is skipped.
echo ""
echo "Generating self-signed TLS certificate ..."
mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_DIR/server.crt" ]; then
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout  "$CERT_DIR/server.key" \
        -out     "$CERT_DIR/server.crt" \
        -days    3650 \
        -subj    "/CN=10.0.0.1" \
        -addext  "subjectAltName=IP:10.0.0.1,IP:100.64.0.1,IP:127.0.0.1"
    echo "  Certificate: $CERT_DIR/server.crt"
    echo "  Key:         $CERT_DIR/server.key"
else
    echo "  Certificate already exists — skipping generation."
fi

# ── 3. Create test file ───────────────────────────────────────────────────────
echo ""
echo "Creating ${FILE_SIZE_MB}MB test file ..."
mkdir -p "$WEB_ROOT"

if [ ! -f "$WEB_ROOT/10MB.zip" ]; then
    dd if=/dev/urandom bs=1M count="$FILE_SIZE_MB" 2>/dev/null \
        | gzip > "$WEB_ROOT/10MB.zip"
    SIZE=$(du -h "$WEB_ROOT/10MB.zip" | cut -f1)
    echo "  Created: $WEB_ROOT/10MB.zip  ($SIZE)"
else
    echo "  Test file already exists: $WEB_ROOT/10MB.zip"
fi

# ── 4. Write Caddyfile ────────────────────────────────────────────────────────
echo ""
echo "Writing Caddyfile ..."

cat > "$CADDY_FILE" << 'EOF'
{
    # Global options
    # Disable automatic HTTPS redirect — we handle TLS manually below
    auto_https off
    # Admin API disabled for simplicity
    admin off
}

# QUIC / HTTP3 server on port 443
:443 {
    tls /etc/caddy/certs/server.crt /etc/caddy/certs/server.key

    # Enable HTTP/3 (QUIC) — Caddy enables this automatically when TLS is set.
    # The Alt-Svc header tells clients that QUIC is available.
    header Alt-Svc 'h3=":443"; ma=86400'

    # Serve files from the web root
    root * /var/www/quic
    file_server browse
}

EOF

mkdir -p /var/log/caddy
echo "  Caddyfile: $CADDY_FILE"

# ── 5. Start Caddy ────────────────────────────────────────────────────────────
echo ""
echo "Starting Caddy ..."

# Stop any existing instance
systemctl stop caddy 2>/dev/null || caddy stop 2>/dev/null || true
sleep 1

# Start
systemctl start caddy 2>/dev/null || caddy start --config "$CADDY_FILE"
sleep 2

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Verification"
echo "=================================================="

# Check caddy is running
if pgrep -x caddy > /dev/null; then
    echo "  ✓ Caddy is running (PID: $(pgrep -x caddy))"
else
    echo "  ✗ Caddy is NOT running — check /var/log/caddy/ for errors"
    exit 1
fi

# Check port 443 is open
if ss -tlnp | grep -q ':443'; then
    echo "  ✓ Port 443 is listening"
else
    echo "  ✗ Port 443 not listening"
fi

# Test download (will fail certificate check without --no-verify)
echo ""
echo "  Testing QUIC download (requires quiche-client on PATH) ..."
if command -v ../../quiche/target/release/examples/http3-client &>/dev/null; then
    if ../../quiche/target/release/examples/http3-client --no-verify \
                                                     --wire-version 00000001 \
                                                     --dump-responses /dev/null \
                                                     "https://10.0.0.1/10MB.zip" 2>&1 | grep -q "200"; then
        echo "  ✓ QUIC download test passed"
    else
        echo "  ✗ QUIC download test failed — check caddy logs"
        echo "    Try manually: quiche-client --no-verify https://10.0.0.1/10MB.zip"
    fi
else
    echo "  quiche-client not found — skipping download test."
    echo "  Build it with:"
    echo "    git clone --recursive https://github.com/cloudflare/quiche"
    echo "    cd quiche && cargo build --release --examples"
    echo "    sudo cp target/release/examples/http3-client /usr/local/bin/quiche-client"
fi

echo ""
echo "=================================================="
echo "  Setup complete. You can now run:"
echo "    bash generate_dataset_quic.sh"
echo "=================================================="