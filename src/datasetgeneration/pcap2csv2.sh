#!/bin/bash
# pcap2csv.sh — extract TCP fields including window size for BiF vs cwnd comparison

file=$1

sudo tshark -r "$file" -Y "tcp" \
  -T fields \
  -e frame.time_relative \
  -e frame.time_delta \
  -e ip.src \
  -e tcp.len \
  -e tcp.seq \
  -e tcp.ack \
  -e tcp.window_size \
  -e tcp.analysis.ack_rtt \
  -E header=y -E separator=, \
  > "$file-tcp.csv" 2>/dev/null

sudo chown $USER:$USER "$file-tcp.csv" 2>/dev/null