python values_creation.py

#Déployer prometheus dans chaque cluster qu'on veut observer 
observer_key="c0"
for key in $(yq eval 'keys | .[]' ../../config/clus_params.yaml); do
    if [ "$key" != "$observer_key" ]; then
        helm install kube-prometheus-$key  --set prometheus.thanos.create=true --set prometheus.externalLabels.cluster="cluster-$key" --namespace monitoring --create-namespace --kube-context kind-$key bitnami/kube-prometheus
        subctl export service kube-prometheus-$key-prometheus --namespace monitoring --context kind-$key
        subctl export service kube-prometheus-$key-alertmanager --namespace monitoring --context kind-$key
        subctl export service kube-prometheus-$key-prometheus-thanos --namespace monitoring --context kind-$key
    
    fi
done

#Attendre que tous les pods du dernier cluster soient prêts  
kubectl wait --for=condition=Ready --timeout=600s pod -A --all
#Déployer thanos dans le cluster observateur ($observer_key)
config_path=../../config/thanos_configuration/thanos_values.yaml
helm install thanos  --values $config_path bitnami/thanos --namespace monitoring --create-namespace --kube-context kind-$observer_key
kubectl wait --for=condition=Ready --timeout=600s pod -A --all
export SERVICE_PORT=$(kubectl get --namespace monitoring -o jsonpath="{.spec.ports[0].port}" services thanos-query --context kind-$observer_key)
kubectl port-forward --namespace monitoring svc/thanos-query ${SERVICE_PORT}:${SERVICE_PORT} --context kind-$observer_key 