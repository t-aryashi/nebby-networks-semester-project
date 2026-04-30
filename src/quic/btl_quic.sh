#!/bin/bash
# btl_quic.sh
#
# Mirror of btl.sh — runs inside the outer mm-delay shell.
# Starts tcpdump (UDP filter), then launches mm-link + inner mm-delay
# + client_quic.sh.
#
# DIFFERENCES FROM btl.sh
# ────────────────────────
#  1. tcpdump filter is "udp" instead of the default (all traffic).
#     QUIC runs over UDP so we only need UDP packets.
#     Snap length (-s 96) captures Ethernet+IP+UDP headers only —
#     the QUIC payload is encrypted anyway and not needed for BiF.
#
#  2. Calls client_quic.sh instead of client.sh
#     (client_quic.sh runs quiche-client instead of wget).
#
# ARGUMENTS (same positions as btl.sh)
#   $1  dump        — output pcap filename (e.g. test_quic.pcap)
#   $2  postdelay   — inner mm-delay in ms
#   $3  buff        — buffer size in bytes
#   $4  aqm         — AQM type (droptail / pfifo)
#   $5  cc          — CCA name (forwarded to client_quic.sh for --cc flag)
#   $6  link        — QUIC URL to download

set -euo pipefail

dump=$1
postdelay=$2
buff=$3
aqm=$4
cc=$5
link=$6

# ── Start tcpdump with UDP filter ─────────────────────────────────────────────
# -i ingress  : Mahimahi's inner interface (same as btl.sh)
# -s 96       : snap 96 bytes — enough for Ethernet(14)+IP(20)+UDP(8)+QUIC hdr
#               The QUIC payload is encrypted; we only need packet sizes
#               and timestamps for compute_bif_quic().
# -q          : quiet (no per-packet output to stdout)
# udp         : BPF filter — only capture UDP (QUIC) packets
tcpdump -i ingress \
        -s 96 \
        -w "$dump" \
        -q \
        udp &
DUMP_PID=$!

# Give tcpdump a moment to open the interface before traffic starts
sleep 0.3

# ── Mahimahi bottleneck + inner delay + QUIC client ───────────────────────────
# Identical nesting to btl.sh: mm-link shapes bandwidth, inner mm-delay
# adds server-side latency, client_quic.sh downloads the target file.
mm-link ../traces/bw.trace ../traces/bw.trace \
    --uplink-queue="$aqm"   --uplink-queue-args="bytes=$buff" \
    --downlink-queue="$aqm" --downlink-queue-args="bytes=$buff" \
    mm-delay "$postdelay" \
    timeout 90 bash client_quic.sh "$cc" "$link" || echo "  WARNING: client timed out"

# ── Stop capture ──────────────────────────────────────────────────────────────
sleep 1
kill "$DUMP_PID" 2>/dev/null || true
wait "$DUMP_PID" 2>/dev/null || true

# Sanity check
if [ -f "$dump" ] && [ -s "$dump" ]; then
    SIZE=$(du -h "$dump" | cut -f1)
    echo "  pcap saved: $dump  ($SIZE)"
else
    echo "  WARNING: $dump is missing or empty."
    echo "    Check that:"
    echo "      1. quiche-client connected successfully"
    echo "      2. The QUIC server is running on the host"
    echo "      3. tcpdump had permission (run as root or with CAP_NET_RAW)"
fi