for key in $(yq eval 'keys | .[]' ../../config/clus_params.yaml); do
    kubectl apply -f ../../multus-cni/deployments/multus-daemonset-thick.yml --context kind-$key
done