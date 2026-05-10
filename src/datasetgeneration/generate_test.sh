#!/bin/bash
CC="reno"
DELAY=50
BW=200
BUFF_MUL=20
AQM="droptail"
URL="https://www.google.com/"   # ← already correct! just confirm this line
mkdir -p ../dataset
mkdir -p ../traces

echo "------------------------------------------------"
echo "GENERATE: CC=$CC Delay=$DELAY BW=$BW Buf=2xBDP"

# Set CCA on the host
sudo sysctl -w net.ipv4.tcp_congestion_control=$CC > /dev/null

./run_test.sh $CC $DELAY $DELAY $BW $BUFF_MUL $URL $AQM
