#!/bin/bash
# test_real_site.sh
# Usage: ./test_real_site.sh <url> <delay_ms>
# Example: ./test_real_site.sh https://www.google.com 50

URL=$1
DELAY=${2:-50}
BW=2000
BUFF_MUL=20
AQM=droptail
CC=external   # label only — we don't control the server's CCA

mkdir -p ../candidates-measurements ../traces

# Generate trace if needed
python3 -c "
bw=$BW
pps = bw*1000//8//1500
with open('../traces/bw_${BW}.trace','w') as f:
    last=0
    for t in range(60):
        for i in range(pps):
            ct = t*1000 + i*1000//pps
            ct = max(ct, last+1)
            f.write(str(ct)+'\n')
            last=ct
"

ts=$(date +%s)
PCAP=/tmp/real_site_${ts}.pcap

echo "Testing: $URL"
echo "Delay: ${DELAY}ms  BW: ${BW}Kbps"

mm-delay $DELAY \
mm-link ../traces/bw_${BW}.trace ../traces/bw_${BW}.trace -- \
    bash -c "
        tcpdump -i ingress -w $PCAP -q 2>/dev/null &
        DPID=\$!
        sleep 0.3
        wget --tries=1 --timeout=30 '$URL' -O /dev/null -q
        sleep 1
        kill \$DPID 2>/dev/null
    "

echo "Converting pcap to CSV..."
sudo tshark -r $PCAP -Y tcp \
    -T fields \
    -e frame.time_relative \
    -e frame.time_delta \
    -e ip.src \
    -e tcp.len \
    -e tcp.seq \
    -e tcp.ack \
    -e tcp.window_size \
    -E header=y -E separator=, \
    > /tmp/real_site_${ts}_tcp.csv 2>/dev/null

echo "CSV saved to: /tmp/real_site_${ts}_tcp.csv"
echo ""
echo "Classifying..."
cd ../nebby
python3 classify.py /tmp/real_site_${ts}_tcp.csv