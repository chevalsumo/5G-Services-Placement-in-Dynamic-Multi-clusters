echo "Configuration du mirror registry Orange sur tous les noeuds crées" 
for node in $(kind get nodes -A); do
    #Pour chaque node de nos clusters
    echo ${node}
    docker cp ./node_repo_config.sh $node:/node_repo_config.sh
    docker cp ../../config/cni_plugins/. $node:/opt/cni/bin/
    docker exec $node /bin/bash /node_repo_config.sh
done