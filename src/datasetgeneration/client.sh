#!/bin/bash
cc=$1
link=$2

# Inside Mahimahi, the host is always reachable at 100.64.0.1
wget --tries=1 --timeout=20 "$link" -O /dev/null