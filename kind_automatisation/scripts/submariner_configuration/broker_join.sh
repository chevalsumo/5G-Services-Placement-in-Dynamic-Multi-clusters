#!/bin/bash

export PATH=$PATH:~/.local/bin
kubectl annotate node c9-worker gateway.submariner.io/public-ip=ipv4:172.18.0.7 --context c9
kubectl label node c9-worker submariner.io/gateway=true --context c9
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c9 --context c9
kubectl annotate node c10-worker gateway.submariner.io/public-ip=ipv4:172.18.0.9 --context c10
kubectl label node c10-worker submariner.io/gateway=true --context c10
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c10 --context c10