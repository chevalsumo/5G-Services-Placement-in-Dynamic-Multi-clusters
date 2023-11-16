
#Générer les fichiers de configurations pour la création des clusters 
python ./cluster_creation.py

for key in $(yq eval 'keys | .[]' ../../config/clus_params.yaml); do
    config_path="../../config/cls_creation/kclust-$key.yaml"
    kind create cluster --config "$config_path" -v 1
done

