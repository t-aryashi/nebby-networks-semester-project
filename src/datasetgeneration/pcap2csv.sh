#!/bin/bash
file=$1

sudo tshark -r "$file" -Y "tcp" \
  -T fields \
  -e frame.time_relative \
  -e frame.time_delta \
  -e ip.src \
  -e tcp.len \
  -e tcp.seq \
  -e tcp.ack \
  -e tcp.analysis.ack_rtt \
  -E header=y -E separator=, \
  > "$file-tcp.csv" 2>/dev/null

sudo chown xcoder:xcoder "$file-tcp.csv"