#!/bin/bash

export PATH=$PATH:~/.local/bin
kubectl config set-cluster c0 --server https://172.18.0.9:6443
subctl deploy-broker
kubectl annotate node c0-worker gateway.submariner.io/public-ip=ipv4:172.18.0.8
kubectl label node c0-worker submariner.io/gateway=true
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid kind-c0
kubectl annotate node c1-worker gateway.submariner.io/public-ip=ipv4:172.18.0.10 --context c1
kubectl label node c1-worker submariner.io/gateway=true --context c1
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c1 --context c1
kubectl annotate node c2-worker gateway.submariner.io/public-ip=ipv4:172.18.0.12 --context c2
kubectl label node c2-worker submariner.io/gateway=true --context c2
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c2 --context c2
kubectl annotate node c3-worker gateway.submariner.io/public-ip=ipv4:172.18.0.14 --context c3
kubectl label node c3-worker submariner.io/gateway=true --context c3
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c3 --context c3
kubectl annotate node c4-worker gateway.submariner.io/public-ip=ipv4:172.18.0.16 --context c4
kubectl label node c4-worker submariner.io/gateway=true --context c4
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c4 --context c4
kubectl annotate node c5-worker gateway.submariner.io/public-ip=ipv4:172.18.0.19 --context c5
kubectl label node c5-worker submariner.io/gateway=true --context c5
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c5 --context c5
kubectl annotate node c6-worker gateway.submariner.io/public-ip=ipv4:172.18.0.21 --context c6
kubectl label node c6-worker submariner.io/gateway=true --context c6
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c6 --context c6
kubectl annotate node c7-worker gateway.submariner.io/public-ip=ipv4:172.18.0.23 --context c7
kubectl label node c7-worker submariner.io/gateway=true --context c7
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c7 --context c7
kubectl annotate node c8-worker gateway.submariner.io/public-ip=ipv4:172.18.0.25 --context c8
kubectl label node c8-worker submariner.io/gateway=true --context c8
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c8 --context c8
kubectl annotate node c9-worker gateway.submariner.io/public-ip=ipv4:172.18.0.27 --context c9
kubectl label node c9-worker submariner.io/gateway=true --context c9
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c9 --context c9
kubectl annotate node c10-worker gateway.submariner.io/public-ip=ipv4:172.18.0.28 --context c10
kubectl label node c10-worker submariner.io/gateway=true --context c10
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c10 --context c10
kubectl annotate node c11-worker gateway.submariner.io/public-ip=ipv4:172.18.0.30 --context c11
kubectl label node c11-worker submariner.io/gateway=true --context c11
subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid c11 --context c11