#!/bin/bash
CCS=("reno" "cubic" "bbr" "bic" "htcp" "hybla" "illinois" "scalable" "vegas" "veno" "westwood" "yeah")
# CCS=("veno" "westwood" "yeah")
# CCS=("vegas" "veno")
# Removed: highspeed, lp, nv, cdg, dctcp
# Reason: these need special conditions (ECN, competing traffic,
#         high cwnd thresholds) not present in the setup
DELAYS=(50 100)
BW=2000
BUFF_MUL=20
AQM="droptail"
URL="http://10.0.0.1/10MB.zip"   # just confirm this line
mkdir -p ../dataset
mkdir -p ../traces

for cc in "${CCS[@]}"; do
  for delay in "${DELAYS[@]}"; do
    echo "------------------------------------------------"
    echo "GENERATE: CC=$cc Delay=$delay BW=$BW Buf=2xBDP"

    # Set CCA on the host
    sudo sysctl -w net.ipv4.tcp_congestion_control=$cc > /dev/null

    ACTUAL=$(cat /proc/sys/net/ipv4/tcp_congestion_control)
    if [ "$ACTUAL" != "$cc" ]; then
        echo "WARNING: $cc not available, got $ACTUAL — SKIPPING"
        continue
    fi
    ./run_test.sh $cc $delay $delay $BW $BUFF_MUL $URL $AQM
  done
done