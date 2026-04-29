#!/bin/bash
# test_real_site.sh — Classify the CCA of a real website
# Usage: ./test_real_site.sh <url> [delay_ms]
# Example: ./test_real_site.sh https://www.google.com 50

URL=$1
DELAY=${2:-50}
BW=2000
AQM=droptail

if [ -z "$URL" ]; then
    echo "Usage: ./test_real_site.sh <url> [delay_ms]"
    echo "Example: ./test_real_site.sh https://www.google.com 50"
    exit 1
fi

mkdir -p ../traces

# Generate bw trace if needed
TRACE=../traces/bw_${BW}.trace
if [ ! -f "$TRACE" ]; then
    python3 -c "
bw = $BW
pps = bw*1000//8//1500
if pps < 1: pps = 1
with open('$TRACE','w') as f:
    last = 0
    for t in range(60):
        for i in range(pps):
            ct = t*1000 + i*1000//pps
            ct = max(ct, last+1)
            f.write(str(ct)+'\n')
            last = ct
"
    echo "Generated trace: $TRACE"
fi

ts=$(date +%s)
PCAP_50=/tmp/real_${ts}_50ms.pcap
PCAP_100=/tmp/real_${ts}_100ms.pcap
CSV_50=/tmp/real_${ts}_50ms_tcp.csv
CSV_100=/tmp/real_${ts}_100ms_tcp.csv

run_capture() {
    local delay=$1
    local pcap=$2
    local csv=$3

    echo ""
    echo "--- Capturing with delay=${delay}ms ---"

    mm-delay $delay \
    mm-link $TRACE $TRACE -- \
        bash -c "
            sudo tcpdump -i ingress -w $pcap -q 2>/dev/null &
            DPID=\$!
            sleep 0.3
            wget --tries=1 --timeout=40 '$URL' -O /dev/null -q 2>/dev/null
            sleep 1
            kill \$DPID 2>/dev/null
            wait \$DPID 2>/dev/null
        "

    if [ ! -f "$pcap" ] || [ ! -s "$pcap" ]; then
        echo "ERROR: No pcap produced at delay=${delay}ms"
        return 1
    fi

    echo "pcap: $(du -h $pcap | cut -f1)"
    echo "Converting to CSV..."

    sudo tshark -r $pcap -Y tcp \
        -T fields \
        -e frame.time_relative \
        -e frame.time_delta \
        -e ip.src \
        -e tcp.len \
        -e tcp.seq \
        -e tcp.ack \
        -e tcp.window_size \
        -E header=y -E separator=, \
        > $csv 2>/dev/null

    local rows=$(wc -l < $csv)
    echo "CSV rows: $rows  →  $csv"
    return 0
}

echo "========================================"
echo "  Testing: $URL"
echo "  BW: ${BW} Kbps"
echo "========================================"

# Run both delay profiles (needed for 6D dual-profile model)
run_capture 50  $PCAP_50  $CSV_50  || exit 1
run_capture 100 $PCAP_100 $CSV_100 || exit 1

echo ""
echo "========================================"
echo "  Classifying (dual profile — 6D)..."
echo "========================================"
cd ../nebby
python3 classify.py $CSV_50 $CSV_100

echo ""
echo "Temp files: $CSV_50  $CSV_100"