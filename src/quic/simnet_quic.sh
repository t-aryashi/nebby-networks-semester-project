#!/bin/bash
# simnet_quic.sh
#
# Mirror of simnet.sh — generates the Mahimahi trace and launches the
# bottleneck shell for a QUIC measurement.
#
# IDENTICAL TO simnet.sh EXCEPT:
#  • Calls btl_quic.sh instead of btl.sh
#  • Writes to test_quic.pcap instead of test.pcap
#
# All buffer math, trace generation, and mm-delay nesting are unchanged.
# The BDP/buffer calculation does not depend on the transport protocol.
#
# ARGUMENTS (same positions as simnet.sh)
#   $1  cc          — CCA name
#   $2  predelay    — outer mm-delay (ms)
#   $3  postdelay   — inner mm-delay (ms)
#   $4  bw          — link speed (Kbps)
#   $5  buff_mul    — buffer multiplier (buff = BDP × buff_mul/10)
#   $6  link        — QUIC target URL
#   $7  aqm         — AQM type

set -euo pipefail

cc=$1
predelay=$2
postdelay=$3
bw=$4
buff_mul=$5
link=$6
aqm=$7

# ── Buffer size calculation (identical to simnet.sh) ─────────────────────────
rtt=$(( predelay + postdelay ))
bw_bytes=$(( bw * 1000 / 8 ))
bdp=$(( bw_bytes * rtt / 1000 ))

# Fixed-point multiply: buff_mul=20 → 2.0× BDP
buff=$(( (bdp * buff_mul) / 10 ))

# Minimum one packet
if [ "$buff" -lt 1500 ]; then
    buff=1500
fi

echo "  RTT=${rtt}ms  BDP=${bdp}B  buffer=${buff}B  AQM=${aqm}"

# ── Mahimahi trace generation (identical to simnet.sh) ───────────────────────
trace=../traces/bw.trace
rm -f "$trace"
touch "$trace"

pps=$(( bw_bytes / 1500 ))
if [ "$pps" -le 0 ]; then pps=1; fi

duration=60   # seconds — same as TCP setup

last_t=0
for (( t=0; t<duration; t++ )); do
    for (( i=0; i<pps; i++ )); do
        curr_t=$(( t * 1000 + i * 1000 / pps ))
        if [ "$curr_t" -le "$last_t" ]; then
            curr_t=$(( last_t + 1 ))
        fi
        echo "$curr_t" >> "$trace"
        last_t=$curr_t
    done
done

echo "  Trace: $trace  (pps=$pps, duration=${duration}s)"

# ── Launch Mahimahi shell (same nesting as simnet.sh) ─────────────────────────
# Outer mm-delay = predelay (client-side latency)
# Inner mm-delay = postdelay (server-side latency)  ← inside btl_quic.sh
#
# Note: test_quic.pcap is passed as the dump path.
# btl_quic.sh starts tcpdump with a UDP filter and then runs
# mm-link + mm-delay + client_quic.sh.

mm-delay "$predelay" bash btl_quic.sh test_quic.pcap "$postdelay" "$buff" "$aqm" "$cc" "$link"