import subprocess
import yaml


def create_ippool_cross_clusters(clus_1: str, clus_2: str) -> None:
    """
    Create an IPPool of one cluster in the other and vice versa.

    Args:
        clus_1 (str): Name of the first cluster.
        clus_2 (str): Name of the second cluster.

    Raises:
        subprocess.CalledProcessError: If calicotl command fails.
    """
    # Create IPPool of cluster 1 in cluster 2 
    config_path = f"../../config/ippools_configuration/ippool_config_{clus_1}.yaml"
    subprocess.run(["calicotl", f"--context=kind-{clus_2}", "create", "-f", config_path, "--skip-exists"], check=True)

    # Create IPPool of cluster 2 in cluster 1 
    config_path = f"../../config/ippools_configuration/ippool_config_{clus_2}.yaml"
    subprocess.run(["calicotl", f"--context=kind-{clus_1}", "create", "-f", config_path, "--skip-exists"], check=True)
    

def create_ippool_cross_all(clus_1: str) -> None: 
    """
    Create an IPPool of one cluster in all the other clusters and vice versa.

    Args:
        clus_1 (str): Name of the cluster.

    Raises:
        subprocess.CalledProcessError: If calicotl command fails.
    """   
    with open('../../config/clus_params.yaml', 'r') as f:
        keys = yaml.safe_load(f).keys()
    
    for clus_2 in keys:
        if clus_2 != clus_1:
            create_ippool_cross_clusters(clus_2, clus_1)

def create_cluster_ippool(clus_name, pod_subnet, service_subnet):
    """
    Creates IP pool configuration for a given cluster.
    Args:
        clus_name (str): Name of the cluster.
        pod_subnet (str): CIDR block for pods in the cluster.
        service_subnet (str): CIDR block for services in the cluster.
    Returns:
        None
    """
    ip_pools = [
        {
            "apiVersion": "projectcalico.org/v3",
            "kind": "IPPool",
            "metadata": {
                "name": f"svc-{clus_name}"
            },
            "spec": {
                "cidr": service_subnet,
                "natOutgoing": False,
                "disabled": True
            }
        },
        {
            "apiVersion": "projectcalico.org/v3",
            "kind": "IPPool",
            "metadata": {
                "name": f"pod-{clus_name}"
            },
            "spec": {
                "cidr": pod_subnet,
                "natOutgoing": False,
                "disabled": True
            }
        }
    ]
    with open(f"../../config/ippools_configuration/ippool_config_{clus_name}.yaml", 'w') as f:
        yaml.dump_all(ip_pools, f)


if __name__ == '__main__':
    with open('../../config/clus_params.yaml', 'r') as f:
        data = yaml.safe_load(f)

    for clus_2 in data:
        pod_subnet =    data[clus_2]['pod_subnet']
        service_subnet = data[clus_2]['service_subnet']
        create_cluster_ippool(clus_2, pod_subnet, service_subnet)