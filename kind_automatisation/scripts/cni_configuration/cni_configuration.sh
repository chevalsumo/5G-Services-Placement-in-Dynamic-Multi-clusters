#Générer les fichiers de configurations CNI à l'aide du script python 
python cni_configuration.py

for key in $(yq eval 'keys | .[]' ../../config/clus_params.yaml); do
    config_path="../../config/cni_configuration/cni_config_$key.yaml"
    kubectx kind-$key
    kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.25.0/manifests/tigera-operator.yaml 
    kubectl apply -f $config_path
    kubectl wait --for=condition=Ready --timeout=600s pod -A --all
done


