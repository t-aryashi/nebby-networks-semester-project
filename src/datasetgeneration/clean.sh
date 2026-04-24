#!/bin/bash
sudo killall -9 tcpdump 2>/dev/null
rm -f test.pcap*
rm -f index*
rm -f upload_data