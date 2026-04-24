#!/bin/bash
# run_test.sh

cc=$1
predelay=$2
postdelay=$3
linkspeed=$4
buff_mul=$5
url=$6
aqm=$7

# 1. Cleanup
rm -f test.pcap* index*

# 2. Run simulation
./simnet.sh $cc $predelay $postdelay $linkspeed $buff_mul $url $aqm

# 3. Check and Convert
if [ -f test.pcap ] && [ -s test.pcap ]; then
    ./pcap2csv.sh test.pcap
    
    outdir="../candidates-measurements"
    mkdir -p $outdir
    
    ts=$(date +%s)
    # The filename IS the label for your ML model
    label="cc-${cc}_aqm-${aqm}_bw-${linkspeed}_buf-${buff_mul}"
    cp test.pcap-tcp.csv "$outdir/${label}_${ts}_tcp.csv"
    echo "Successfully saved: ${label}_${ts}_tcp.csv"
else
    echo "Error: Simulation failed to produce pcap"
fi