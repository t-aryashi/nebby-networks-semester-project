#!/bin/bash
# btl.sh

dump=$1
postdelay=$2
buff=$3
aqm=$4
cc=$5
link=$6

# START CAPTURE WITH SUDO INTERNALLY
tcpdump -i ingress -w "$dump" -q &
DUMP_PID=$!

mm-link ../traces/bw.trace ../traces/bw.trace \
    --uplink-queue=$aqm --uplink-queue-args="bytes=$buff" \
    --downlink-queue=$aqm --downlink-queue-args="bytes=$buff" \
    mm-delay $postdelay ./client.sh $cc $link

# Stop tcpdump
sleep 1
kill $DUMP_PID 2>/dev/null