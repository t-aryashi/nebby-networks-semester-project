#!/usr/bin/env python3
"""
selenium_server.py — Multi-asset HTTP server for Selenium measurements
Paper reference: Nebby §3.5, §4.5

The paper measures real websites and correlates flows to asset types:
  - Video chunks    → usually BBR
  - Static assets   → usually CUBIC
  - Audio streams   → usually BBR

Since we can't hit real websites, we simulate this by serving multiple
file types from a local server. The key point is that Chrome will open
MULTIPLE CONCURRENT TCP CONNECTIONS to fetch different assets — and each
connection can potentially use a different CCA (set by sysctl on the server).

This server serves:
  /                     → index.html (the page Chrome loads)
  /video.bin            → 5MB binary file  (simulates video chunk)
  /style.css            → 200KB file       (simulates static asset)
  /script.js            → 200KB file       (simulates static asset)
  /image1.bin           → 500KB file       (simulates image)
  /image2.bin           → 500KB file       (simulates image)
  /font.bin             → 100KB file       (simulates font)

Usage:
    sudo python3 selenium_server.py [--port 8080] [--dir /path/to/assets]
"""

import os
import sys
import argparse
import http.server
import socketserver
import threading

ASSET_DIR  = '/tmp/nebby_assets'
DEFAULT_PORT = 8080


def create_assets(asset_dir):
    """Create dummy files of realistic sizes for each asset type."""
    os.makedirs(asset_dir, exist_ok=True)

    assets = {
        'video.bin':   20 * 1024 * 1024,   # 20MB
        'style.css':   2 * 1024 * 1024,    # 2MB
        'script.js':   2 * 1024 * 1024,
        'image1.bin':  3 * 1024 * 1024,
        'image2.bin':  3 * 1024 * 1024,
        'font.bin':    1 * 1024 * 1024,
    }

    for fname, size in assets.items():
        fpath = os.path.join(asset_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Creating {fname} ({size//1024} KB)...")
            with open(fpath, 'wb') as f:
                f.write(os.urandom(size))

    # HTML page that loads all assets — Chrome will open parallel connections
    html = """<!DOCTYPE html>
<html>
<head>
  <title>Nebby Selenium Test Page</title>
  <link rel="stylesheet" href="/style.css">
  <script src="/script.js"></script>
</head>
<body>
  <h1>Nebby Browser Measurement Page</h1>
  <p>This page loads multiple assets to generate concurrent TCP flows.</p>

  <!-- Images — static assets -->
  <img src="/image1.bin" alt="img1" style="display:none">
  <img src="/image2.bin" alt="img2" style="display:none">

  <!-- Font — static asset -->
  <style>
    @font-face { font-family: 'TestFont'; src: url('/font.bin'); }
  </style>

  <!-- Video — the "heavy" flow, simulates video streaming -->
  <video id="player" style="display:none">
    <source src="/video.bin" type="video/mp4">
  </video>

  <script>
    // Trigger video load after page load
    window.onload = function() {
      document.getElementById('player').load();
    };
  </script>
</body>
</html>"""

    with open(os.path.join(asset_dir, 'index.html'), 'w') as f:
        f.write(html)

    print(f"  Assets created in {asset_dir}")
    return assets


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves from asset_dir and suppresses access logs."""
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        pass   # suppress request logs


def start_server(port=DEFAULT_PORT, asset_dir=ASSET_DIR):
    """Start HTTP server in a background thread. Returns the server object."""
    handler = lambda *args, **kwargs: QuietHTTPHandler(
        *args, directory=asset_dir, **kwargs)

    server = socketserver.TCPServer(('0.0.0.0', port), handler)
    server.allow_reuse_address = True

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    print(f"Server running on http://0.0.0.0:{port}")
    print(f"Serving from: {asset_dir}")
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--dir',  default=ASSET_DIR)
    args = parser.parse_args()

    print("=== Creating assets ===")
    assets = create_assets(args.dir)

    print("\n=== Starting server ===")
    server = start_server(args.port, args.dir)

    print(f"\nServer ready. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()