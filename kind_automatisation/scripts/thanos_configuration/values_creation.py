import yaml
import subprocess
from typing import List
import sys
import re 
import shlex

def  get_deployed_services_names(monitoring_ns: str, cluster: str) -> List[str]:
    """
    Gets the names of the deployed Thanos services in a given monitoring namespace and cluster.

    Args:
        monitoring_ns (str): The monitoring namespace in which Thanos services are deployed.
        cluster (str): The name of the cluster being monitored.

    Returns:
        List[str]: A list of the names of the deployed Thanos services.
    """
    # Use subprocess to run the kubectl command to get the list of deployed services in the monitoring namespace
    result = subprocess.run(["kubectl", "get", "services", "-n", monitoring_ns, "--context", f"kind-{cluster}", "--template", "'{{range .items}}{{.metadata.name}}{{\"\\n\"}}{{end}}'"], capture_output=True, text=True)
    
    # Parse the result and extract the names of the Thanos services the we wanna deploy
    services = []
    for line in result.stdout.split('\n'):
        match = re.search(r'(prometheus-thanos|prometheus|alertmanager)$', line)
        if match:
            print(line)
            pattern = r".*pth-c" 
            match_corr = re.search(pattern, line)
            if match_corr: 
                matched_string = match_corr.group()
                line = line.replace(matched_string[:-5], "")
                print("corr")
                print(line)
            services.append(line)

    return services

def deploy_prometheus_to_clusters(monitoring_ns, cluster_list, prefix_prths):
    """
    Deploys prometheus to selected clusters and exports services for federation.
    Args:
        monitoring_ns (str): Namespace for monitoring.
        cluster_list (list): List of clusters to deploy to.
        prefix_prths (str): Prefix for kube-prometheus helm chart.
    Returns:
        None
    """
    for key in cluster_list:
        helm_cmd = f"helm upgrade --install {prefix_prths}-{key} --set prometheus.thanos.create=true --set prometheus.externalLabels.cluster=\"cluster-{key}\" --namespace {monitoring_ns} --create-namespace --kube-context kind-{key} bitnami/kube-prometheus"
        subprocess.run(helm_cmd, shell=True, check=True)
        services = get_deployed_services_names(monitoring_ns, key)
        for service in services:
            subctl_cmd = f"subctl export service {service} --namespace {monitoring_ns} --context kind-{key}".replace("'", "ky")
            subprocess.run(subctl_cmd, shell=True, check=True)
    print(services)
def uninstall_prometheus_from_clusters(monitoring_ns, cluster_list, prefix_prths):
    for key in cluster_list:
        helm_cmd = f"helm -n {monitoring_ns} --kube-context kind-{key} uninstall {prefix_prths}-{key}"
        subprocess.run(helm_cmd, shell=True, check=True)
    
def update_thanos_values(namespace: str, prefix_prths: str, clusters: List[str], thanos_value_path: str) -> None:
    """
    Creates a Thanos configuration file based on a given namespace and a list of clusters.

    Args:
        namespace (str): The namespace in which the Thanos components will be installed.
        prefix_prths (str): The prefix of the Prometheus Helm release to be used for Thanos integration.
        clusters (List[str]): The list of clusters for which to generate Thanos configuration.
        thanos_value_path (str): The path where the Thanos values.yaml file will be created.

    Returns:
        None: The function does not return anything, it simply generates a Thanos configuration file.
    """
    config = {
        "objstoreConfig": {
            "type": "s3",
            "config": {
                "bucket": "thanos",
                "endpoint": "{{ include \"thanos.minio.fullname\" . }}.{{ .Release.Namespace }}.svc.cluster.local:9000",
                "access_key": "minio",
                "secret_key": "minio123",
                "insecure": True
            }
        },
        "query": {
            "stores": []
        },
        "bucketweb": {
            "enabled": True
        },
        "compactor": {
            "enabled": True
        },
        "storegateway": {
            "enabled": True
        },
        "ruler": {
            "enabled": True,
            "alertmanagers": [],
            "config": {
                "groups": [
                    {
                        "name": "metamonitoring",
                        "rules": [
                            {
                                "alert": "PrometheusDown",
                                "expr": f'absent(up{{prometheus="{namespace}/{prefix_prths}"}})'
                            }
                        ]
                    }
                ]
            }
        },
        "minio": {
            "enabled": True,
            "auth": {
                "rootPassword": "minio123",
                "rootUser": "minio"
            },
            "monitoringBuckets": "thanos",
            "accessKey": {
                "password": "minio"
            },
            "secretKey": {
                "password": "minio123"
            }
        }
    }
    for key in clusters:
            config["query"]["stores"].append(f"{prefix_prths}-{key}-kube-prometheus-prometheus-thanos.{namespace}.svc.clusterset.local:10901")
            config["ruler"]["alertmanagers"].append(f"http://{prefix_prths}-{key}-kube-prometheus-alertmanager.{namespace}.svc.clusterset.local:9093")

    with open(thanos_value_path, 'w') as f:
        yaml.dump(config, f)


def install_thanos(thanos_value_path: str, observer_key: str, namespace: str) -> None:
    """
    Installs Thanos to the specified cluster using Helm and applies a Thanos configuration.

    Args:
        thanos_value_path (str): The path to the Thanos configuration values file.
        observer_key (str): The key of the observer cluster.
        namespace (str): The namespace in which to install Thanos.

    Returns:
        None: The function does not return anything, it simply installs Thanos.
    """
    helm_cmd = f"helm upgrade --install thanos bitnami/thanos --namespace {namespace} --create-namespace --kube-context kind-{observer_key} --values {thanos_value_path}"
    subprocess.run(helm_cmd, shell=True, check=True)

    kubectl_cmd = f"kubectl wait --for=condition=Ready --timeout=600s pod --all --namespace {namespace} --context kind-{observer_key}"
    #subprocess.run(kubectl_cmd, shell=True, check=True)

    service_port_cmd = f"kubectl get --namespace {namespace} -o jsonpath=\"{{.spec.ports[0].port}}\" services thanos-query --context kind-{observer_key}"
    service_port = subprocess.check_output(service_port_cmd, shell=True, text=True).strip()

    port_forward_cmd = f"kubectl port-forward --namespace {namespace} svc/thanos-query {service_port}:{service_port} --context kind-{observer_key}"
    #subprocess.run(port_forward_cmd, shell=True, check=True)
    print(port_forward_cmd)

if __name__ == '__main__':
    
    cls = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11"]
    if sys.argv[1] == 'deploy_pth': 
        deploy_prometheus_to_clusters("monitoring", cls, "pth")
        #uninstall_prometheus_from_clusters("monitoring","")
        #update_thanos_values("monitoring", "pth", cls, "../../config/thanos_configuration/thanos_values.yaml")
        #install_thanos("../../config/thanos_configuration/thanos_values.yaml", "c0", "monitoring")
    elif sys.argv[1] == "update_values":
        update_thanos_values("monitoring", "pth", cls, "../../config/thanos_configuration/thanos_values.yaml")
    elif sys.argv[1] == "install_thanos":
        install_thanos("../../config/thanos_configuration/thanos_values.yaml", "c0", "monitoring")