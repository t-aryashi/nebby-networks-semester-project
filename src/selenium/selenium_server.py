#!/usr/bin/env python3
"""
selenium_server.py — Multi-asset HTTP server with per-socket CCA control
Paper reference: Nebby §3.5, §4.5

PURPOSE:
  The paper's Table 8 finding: same webpage, different CCAs for different
  content types. Video flows use BBR, static assets use CUBIC. This happens
  because real websites use different CDNs for different content, and each
  CDN independently chooses its CCA.

  We simulate this using Linux's TCP_CONGESTION socket option, which lets
  individual TCP connections override the global sysctl CCA. This means:
    /video.bin   → connection uses BBR   (simulates video CDN)
    /image*.bin  → connection uses BBR   (simulates media CDN)
    /style.css   → connection uses CUBIC (simulates static CDN)
    /script.js   → connection uses CUBIC (simulates static CDN)

  This directly replicates the multi-CCA-per-page behaviour the paper
  observed on real websites like AppleTV and Twitch.

ASSET → CCA MAPPING:
  video.bin    → BBR     (large, long-running → shows BBR probe pattern)
  image1.bin   → BBR     (medium, shows BBR or CUBIC depending on size)
  image2.bin   → CUBIC   (medium, shows CUBIC sawtooth)
  style.css    → CUBIC   (static asset → loss-based CCA)
  script.js    → RENO    (static asset → simulates third CDN)

  This gives you 3 different CCAs across 5 assets — mirrors what the
  paper observed (e.g. AppleTV: BBR for video, CUBIC for ads).

ASSET SIZES (tuned for 2000 Kbps / 5 concurrent flows):
  Each flow gets ~400 Kbps effective bandwidth.
  At 400 Kbps, need 10+ seconds → file must be > 500 KB.
  Use much larger files to ensure multiple oscillation cycles.

Usage:
  # Delete old assets first if they exist:
  rm -rf /tmp/nebby_assets

  # Start server (no sudo needed — port 8080):
  python3 selenium_server.py --port 8080

  # Verify:
  curl http://10.0.0.1:8080/video.bin -o /dev/null -v 2>&1 | head -20
"""

import os
import sys
import socket
import argparse
import http.server
import socketserver
import threading

ASSET_DIR    = '/tmp/nebby_assets'
DEFAULT_PORT = 8080

# Linux socket option for per-connection CCA override
# Defined in linux/tcp.h as TCP_CONGESTION = 13
TCP_CONGESTION = 13

# Asset → CCA mapping
# This simulates different CDNs serving different content types
ASSET_CCA_MAP = {
    'video.bin':  'bbr',    # video CDN uses BBR
    'image1.bin': 'bbr',    # media CDN uses BBR
    'image2.bin': 'cubic',  # image CDN uses CUBIC
    'style.css':  'cubic',  # static CDN uses CUBIC
    'script.js':  'reno',   # JS CDN uses Reno
}

# Asset sizes — large enough for 10+ seconds at 400 Kbps per flow
# 400 Kbps * 10s / 8 = 500 KB minimum → use much more for safety
ASSET_SIZES = {
    'video.bin':  20 * 1024 * 1024,   # 20MB — long video chunk
    'image1.bin':  5 * 1024 * 1024,   # 5MB  — large image
    'image2.bin':  5 * 1024 * 1024,   # 5MB  — large image
    'style.css':   3 * 1024 * 1024,   # 3MB  — CSS bundle
    'script.js':   3 * 1024 * 1024,   # 3MB  — JS bundle
}


def set_socket_cca(sock, cca):
    """
    Set the TCP congestion control algorithm for a specific socket.

    This uses the TCP_CONGESTION socket option (value 13) defined in
    linux/tcp.h. It overrides the global sysctl setting for this one
    connection only — exactly how different CDN servers choose their CCA
    independently of each other.

    Parameters
    ----------
    sock : socket.socket — the TCP connection socket
    cca  : str — CCA name ('bbr', 'cubic', 'reno', etc.)
    """
    try:
        # CCA name must be null-terminated bytes
        cca_bytes = cca.encode('ascii') + b'\x00'
        sock.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cca_bytes)
    except OSError as e:
        # CCA not available in kernel — fall back to kernel default
        print(f"  WARNING: Cannot set CCA={cca} on socket: {e}")


def create_assets(asset_dir):
    """Create dummy asset files of realistic sizes."""
    os.makedirs(asset_dir, exist_ok=True)

    for fname, size in ASSET_SIZES.items():
        fpath = os.path.join(asset_dir, fname)
        if not os.path.exists(fpath):
            cca   = ASSET_CCA_MAP.get(fname, 'unknown')
            print(f"  Creating {fname:<15} ({size//1024//1024}MB) → CCA: {cca}")
            with open(fpath, 'wb') as f:
                # Write in chunks to avoid memory issues for large files
                chunk = 1024 * 1024   # 1MB chunks
                written = 0
                while written < size:
                    to_write = min(chunk, size - written)
                    f.write(os.urandom(to_write))
                    written += to_write

    # HTML page — loads assets sequentially to avoid simultaneous competition
    # Sequential loading ensures each flow gets its fair share of bandwidth
    # and produces a clean BiF trace for classification
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Nebby Selenium Measurement Page</title>
</head>
<body>
  <h1>Nebby — CCA Measurement Page</h1>
  <p>Loading assets to generate TCP flows with different CCAs...</p>
  <div id="status"></div>

  <script>
  // Load assets sequentially so flows don't all compete for bandwidth.
  // Each asset gets its own TCP connection with a specific CCA set
  // server-side using TCP_CONGESTION socket option.
  const assets = [
    {{ url: '/video.bin',  label: 'video (BBR)',    delay: 0    }},
    {{ url: '/image1.bin', label: 'image1 (BBR)',   delay: 500  }},
    {{ url: '/image2.bin', label: 'image2 (CUBIC)', delay: 1000 }},
    {{ url: '/style.css',  label: 'CSS (CUBIC)',    delay: 1500 }},
    {{ url: '/script.js',  label: 'JS (Reno)',      delay: 2000 }},
  ];

  const status = document.getElementById('status');

  async function loadAsset(url, label) {{
    status.innerHTML += `<p>Loading ${{label}}...</p>`;
    try {{
      const response = await fetch(url);
      const buffer   = await response.arrayBuffer();
      status.innerHTML += `<p>✓ ${{label}} — ${{(buffer.byteLength/1024/1024).toFixed(1)}}MB</p>`;
    }} catch(e) {{
      status.innerHTML += `<p>✗ ${{label}} — error: ${{e.message}}</p>`;
    }}
  }}

  async function loadAll() {{
    for (const asset of assets) {{
      await new Promise(r => setTimeout(r, asset.delay));
      await loadAsset(asset.url, asset.label);
    }}
    status.innerHTML += '<p><b>All assets loaded.</b></p>';
  }}

  window.onload = loadAll;
  </script>
</body>
</html>"""

    with open(os.path.join(asset_dir, 'index.html'), 'w') as f:
        f.write(html)

    print(f"\n  Assets created in: {asset_dir}")
    print(f"\n  Asset → CCA mapping:")
    for fname, cca in ASSET_CCA_MAP.items():
        size_mb = ASSET_SIZES[fname] // 1024 // 1024
        print(f"    /{fname:<15} ({size_mb:>2}MB) → {cca}")


class CCAHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP handler that sets a specific CCA per TCP connection
    based on the requested asset path.
    """

    def setup(self):
        """Called before handle(). Set CCA on the socket here."""
        # Determine which CCA to use for this connection
        # We don't know the path yet at setup() time, so we peek at
        # the raw request. As a fallback we set CUBIC (most common).
        # The actual per-path setting happens in do_GET().
        super().setup()

    def do_GET(self):
        """Handle GET request — set CCA based on requested path."""
        # Extract filename from path (strip leading /)
        path = self.path.lstrip('/')
        path = path.split('?')[0]   # remove query params

        # Get CCA for this asset
        cca = ASSET_CCA_MAP.get(path, 'cubic')   # default: cubic

        # Set CCA on this specific TCP connection
        set_socket_cca(self.connection, cca)

        # Log the CCA assignment
        self.log_message(
            f"GET /{path} → CCA={cca}  "
            f"client={self.client_address[0]}:{self.client_address[1]}"
        )

        # Serve the file normally
        super().do_GET()

    def log_message(self, format, *args):
        """Custom log — show CCA assignments, suppress routine GET logs."""
        if 'CCA=' in format:
            print(f"  [server] {format % args}")
        # Suppress normal access logs (too noisy)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ASSET_DIR, **kwargs)


class ReusableTCPServer(socketserver.TCPServer):
    """TCPServer with SO_REUSEADDR to avoid 'address already in use' errors."""
    allow_reuse_address = True


def start_server(port=DEFAULT_PORT, asset_dir=ASSET_DIR):
    """Start the HTTP server. Returns the server object."""
    server = ReusableTCPServer(('0.0.0.0', port), CCAHandler)

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    print(f"\nServer listening on http://0.0.0.0:{port}")
    print(f"Serving from: {asset_dir}")
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nebby multi-asset server with per-socket CCA control'
    )
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to listen on (default: {DEFAULT_PORT})')
    parser.add_argument('--dir',  default=ASSET_DIR,
                        help=f'Asset directory (default: {ASSET_DIR})')
    parser.add_argument('--recreate', action='store_true',
                        help='Delete and recreate all asset files')
    args = parser.parse_args()

    if args.recreate:
        import shutil
        if os.path.exists(args.dir):
            shutil.rmtree(args.dir)
            print(f"Deleted {args.dir}")

    print("=== Creating assets ===")
    create_assets(args.dir)

    print("\n=== Starting server ===")
    server = start_server(args.port, args.dir)

    print("\nPress Ctrl+C to stop.\n")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()