#!/bin/bash
sudo ip netns del srv_ns 2>/dev/null
sudo ip netns del cli_ns 2>/dev/null

sudo ip link del veth_srv 2>/dev/null
sudo ip link del veth_cli 2>/dev/null

sudo pkill -f "http.server" 2>/dev/null
sudo fuser -k 80/tcp 2>/dev/null
echo "Cleaned everything"