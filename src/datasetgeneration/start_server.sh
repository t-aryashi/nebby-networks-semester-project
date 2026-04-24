#!/bin/bash
sudo killall python3 2>/dev/null
sleep 1

if [ ! -f 10MB.zip ]; then
    dd if=/dev/urandom of=10MB.zip bs=1M count=10 status=none
fi

echo "Starting web server on host..."
nohup python3 -m http.server 80 --bind 0.0.0.0 > server.log 2>&1 &

sleep 2

if ss -tulnp | grep -q ":80"; then
    echo "Server is UP on http://localhost:80"
else
    echo "Server failed. Check server.log"
    exit 1
fi