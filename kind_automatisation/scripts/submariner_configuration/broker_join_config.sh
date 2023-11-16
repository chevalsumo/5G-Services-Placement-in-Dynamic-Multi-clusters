key="c0"

python ./broker_context.py
docker cp ../../config/new_broker_config.yaml $key-control-plane:/etc/kubernetes/admin.conf


docker cp ./broker_join.sh $key-control-plane:/broker_join.sh

#Attacher tous les gateways au broker 
docker exec $key-control-plane /bin/bash chmod +x /broker_join.sh
docker exec $key-control-plane /bin/bash /broker_join.sh