#!/bin/bash
# simnet.sh

cc=$1
predelay=$2
postdelay=$3
bw=$4
buff_mul=$5
link=$6
aqm=$7

rtt=$(($predelay + $postdelay))
bw_bytes=$(($bw * 1000 / 8))
bdp=$(($bw_bytes * $rtt / 1000))

# Fixed Math: Multiply then divide to simulate decimals
# If buff_mul is 5, it represents 0.5x BDP
buff=$(( ($bdp * $buff_mul) / 10 ))

# Ensure buffer is at least 1 packet
if [ $buff -lt 1500 ]; then buff=1500; fi

# Generate trace

trace=../traces/bw.trace
rm -f $trace
touch $trace

# packets/sec (~1500B packets)
pps=$(($bw_bytes / 1500))
if [ $pps -le 0 ]; then pps=1; fi

duration=60   # seconds (IMPORTANT)

last_t=0

for ((t=0; t<$duration; t++)); do
  for ((i=0; i<$pps; i++)); do
    curr_t=$((t * 1000 + i * 1000 / pps))

    # enforce strictly increasing timestamps
    if [ $curr_t -le $last_t ]; then
        curr_t=$((last_t + 1))
    fi

    echo $curr_t >> $trace
    last_t=$curr_t
  done
done

mm-delay $predelay ./btl.sh test.pcap $postdelay $buff $aqm $cc $link