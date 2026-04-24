#!/bin/bash
# setup_env.sh - Run with sudo!

echo "Creating namespaces..."
ip netns add srv_ns
ip netns add cli_ns

echo "Creating virtual ethernet link..."
ip link add veth_srv type veth peer name veth_cli

echo "Assigning links to namespaces..."
ip link set veth_srv netns srv_ns
ip link set veth_cli netns cli_ns

echo "Configuring IP addresses and bringing interfaces up..."
# Server Side
ip netns exec srv_ns ip addr add 10.0.0.1/24 dev veth_srv
ip netns exec srv_ns ip link set veth_srv up
ip netns exec srv_ns ip link set lo up

# Client Side
ip netns exec cli_ns ip addr add 10.0.0.2/24 dev veth_cli
ip netns exec cli_ns ip link set veth_cli up
ip netns exec cli_ns ip link set lo up

# Add a default route inside the client namespace so it knows where to send traffic
sudo ip netns exec cli_ns ip route add default dev veth_cli
# Ensure the server knows how to get back to the client
sudo ip netns exec srv_ns ip route add default dev veth_srv

echo "Virtual network ready! Server: 10.0.0.1, Client: 10.0.0.2"