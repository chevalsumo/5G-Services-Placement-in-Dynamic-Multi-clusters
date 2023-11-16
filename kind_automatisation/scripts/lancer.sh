kind get clusters | xargs -t -n1 kind delete cluster --name
#Création des clusters 
cd cls_creation/
./cluster_creation.sh
#Configuration du docker mirror orange
cd ../cls_dockerrepo
./clusters_docker.sh
#
cd ../cni_configuration 
./cni_configuration.sh
#Attente que tous les pods soient prêt pour le déploiment de submariner 
kubectl wait --for=condition=Ready --timeout=600s pod -A --all
cd ../ippools_configuration
./ippools_configuration.sh
cd ../submariner_configuration
./broker_attache_file.sh
./broker_deploy.sh
./broker_join_config.sh
cd ../cni_mutus
./cni_mutus.sh

kubectl wait --for=condition=Ready --timeout=600s pod -A --all
#cd ../thanos_configuration
#./thanos_deployement.sh
