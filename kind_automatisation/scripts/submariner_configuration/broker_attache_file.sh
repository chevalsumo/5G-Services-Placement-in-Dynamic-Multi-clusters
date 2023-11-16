
key="c0"
docker cp ./broker_config.sh $key-control-plane:/broker_config.sh

#Installatino de submariner 
docker exec $key-control-plane /bin/bash /broker_config.sh

