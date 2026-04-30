#!/usr/bin/env bash
# pcap2csv_quic.sh — Extract QUIC/UDP fields from a pcap into a CSV
#
# This is the QUIC counterpart of pcap2csv.sh (which extracts TCP fields).
# quic_bif.py reads the CSV produced here.
#
# FIELDS EXTRACTED
# ─────────────────────────────────────────────────────────────────────────────
#   frame.time_relative   — seconds since first packet in capture
#   ip.src                — source IP (used to split server vs client)
#   ip.dst                — destination IP
#   udp.length            — UDP datagram length INCLUDING 8-byte header
#                           actual payload = udp.length - 8
#
# WHY ONLY UDP.LENGTH AND NOT QUIC FIELDS?
# ─────────────────────────────────────────────────────────────────────────────
# QUIC packets are fully encrypted (RFC 9000 §12.3).  tshark can parse
# QUIC long-header metadata (version, connection ID) but NOT sequence
# numbers or ACK numbers — they live inside the encrypted QUIC payload.
# quic_bif.py's two-assumption model (paper §3.1) works from raw UDP
# packet sizes and directions, so udp.length is all we need.
#
# QUIC FILTER
# ─────────────────────────────────────────────────────────────────────────────
# We filter on UDP port 443 (the standard QUIC port) plus the most
# common alternate ports used by major CDNs:
#   443   — standard HTTPS/QUIC (Cloudflare, Google, Meta, Apple)
#   80    — plain-text QUIC (rare but used by some servers in testing)
#   8443  — alternate TLS/QUIC
#   4433  — quiche default test port
#
# If you know your server's QUIC port, replace the filter with:
#   -Y "udp.port == <your_port>"
# or simply "udp" to capture all UDP (then filter by IP in quic_bif.py).
#
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
#   bash pcap2csv_quic.sh  <input.pcap>  <output_quic.csv>
#
# EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
#   bash pcap2csv_quic.sh  cc-cubic_quic_50ms.pcap  cc-cubic_quic_50ms.csv
#   bash pcap2csv_quic.sh  cc-bbr_quic_100ms.pcap   cc-bbr_quic_100ms.csv
#
# OUTPUT FORMAT (header row + one row per UDP packet)
# ─────────────────────────────────────────────────────────────────────────────
#   "frame.time_relative","ip.src","ip.dst","udp.length"
#   "0.000000","10.0.0.1","10.0.0.2","1252"
#   "0.001234","10.0.0.2","10.0.0.1","41"
#   ...
#
# REQUIREMENTS
# ─────────────────────────────────────────────────────────────────────────────
#   tshark >= 3.0   (sudo apt install tshark  or  brew install wireshark)

set -euo pipefail

# ── argument check ────────────────────────────────────────────────────────────
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input.pcap> <output_quic.csv>"
    echo ""
    echo "Examples:"
    echo "  $0 cc-cubic_quic_50ms.pcap  cc-cubic_quic_50ms.csv"
    echo "  $0 cc-bbr_quic_100ms.pcap   cc-bbr_quic_100ms.csv"
    exit 1
fi

PCAP="$1"
CSV="$2"

if [[ ! -f "$PCAP" ]]; then
    echo "Error: pcap file not found: $PCAP"
    exit 1
fi

# ── QUIC BPF display filter ───────────────────────────────────────────────────
# Captures UDP on all common QUIC ports.
# Extend this list if your server uses a non-standard port.
QUIC_FILTER="udp.port in {443 80 8080 8443 4433}"

echo "Converting: $PCAP → $CSV"
echo "Filter    : $QUIC_FILTER"

tshark \
    -r  "$PCAP" \
    -Y  "$QUIC_FILTER" \
    -T  fields \
    -e  frame.time_relative \
    -e  ip.src \
    -e  ip.dst \
    -e  udp.length \
    -E  header=y \
    -E  separator=, \
    -E  quote=d \
    -E  occurrence=f \
    > "$CSV"

# ── sanity check ──────────────────────────────────────────────────────────────
LINE_COUNT=$(wc -l < "$CSV")
echo "Done. Rows written (including header): $LINE_COUNT"

if [[ $LINE_COUNT -lt 2 ]]; then
    echo ""
    echo "WARNING: Only $LINE_COUNT line(s) in output."
    echo "  Possible causes:"
    echo "    1. No QUIC traffic in pcap (check server is QUIC-capable)"
    echo "    2. Server uses a non-standard port — edit QUIC_FILTER above"
    echo "    3. Capture was too short (< 400 KB transferred)"
    echo ""
    echo "  To capture all UDP and filter by IP later, change QUIC_FILTER to:"
    echo '    QUIC_FILTER="udp"'
fi