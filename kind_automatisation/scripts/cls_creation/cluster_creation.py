import sys
import yaml
import subprocess
import os
import re

sys.path.append('../cni_configuration')
sys.path.append("../submariner_configuration")
sys.path.append("../ippools_configuration")

from cni_configuration import CNI_tigera_install, CNI_multus_install
from broker_context import *
from ippools_configuration import *


def add_clusters(n_clusters, operation = "add"):
    """
    Adds the specified number of clusters.

    Args:
    - n_clusters (int): The number of clusters to add.
    """
    broker_name = "c0"
    if operation == "add" :
        #Load paramter file of created clusters 
        with open('../../config/clus_params.yaml', 'r') as file:
            clus_params = yaml.safe_load(file)
            start_index = len(clus_params)
    else :
        #Create Kind config file for each cluster 
            clus_params = {f"c{i}": {} for i in range(n_clusters)}
            start_index = 0 
    
    clusters = []
    for i in range(start_index, start_index + n_clusters):
        name = f"c{i}"
        clusters.append(name)
        pod_subnet = f"10.{230+i}.0.0/16"
        service_subnet = f"10.{110+i}.0.0/16"
        clus_params[name] = {}
        clus_params[name]["service_subnet"] = service_subnet
        clus_params[name]["pod_subnet"] = pod_subnet
        config = {
            "kind": "Cluster",
            "name": name,
            "apiVersion": "kind.x-k8s.io/v1alpha4",
            "nodes": [
                {"role": "control-plane"},
                {"role": "worker"},
            ],
            "networking": {
                "podSubnet": pod_subnet,
                "serviceSubnet": service_subnet,
                "disableDefaultCNI": True
            }
        }
        config_path = f"../../config/cls_creation/kclust-{name}.yaml"
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        subprocess.run(["kind", "create", "cluster", "--config", config_path, "-v", "1"])
        # Configures the Orange mirror registry on cluster nodes
        configure_mirror_registry(name)
        # CNI configuration 
        CNI_tigera_install(name, pod_subnet)
        CNI_multus_install(name)
        # Extract cluster nodes IPs
        add_cluster_ips(name, save = True)
        # Save the IPPool
        create_cluster_ippool(name, pod_subnet, service_subnet)
        
        # Enable communication with the broker 
        # create_ippool_cross_clusters(name, broker_name)
    with open('../../config/clus_params.yaml', 'w') as file:
        yaml.dump(clus_params, file)
    if operation == "new" :
        # Deply submariner in the broker cluster

        broker_config = "../submariner_configuration/broker_config.sh"
        install_submariner(broker_name, broker_config)
        
    # Submariner Configuration
    # Build the new broker context file 
    build_broker_context(broker_name)
   
    # New Clusters join broker 
    join_broker(broker_name, clusters = clusters, deploy = (operation == "new"))

    # Enable communication with other clusters 
    for clus in clusters : 
        create_ippool_cross_all(clus)
def del_clusters(clusters_to_delete):
    """
    Deletes the specified clusters.

    :param clusters_to_delete: A list of cluster names to delete.
    :type clusters_to_delete: list of str
    """
    # Load the cluster parameters file
    with open('../../config/clus_params.yaml', 'r') as file:
        clus_params = yaml.safe_load(file)

    # Delete each specified cluster
    for cls in clusters_to_delete:
        subprocess.run(['kind', 'delete', 'cluster', '--name', cls])
        # Remove the configuration file for the cluster
        config_path = f"../../config/cls_creation/kclust-{cls}.yaml"
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # Remove the CNI confiugration file 
        config_path = f"../../config/cni_configuration/cni_config_{cls}.yaml"
        # Remove the context file of the cluster if it exists 
        if os.path.exists(config_path):
            os.remove(config_path)

        for file_name in os.listdir('../../config/cls_contexts/'):
            if re.match(f"{cls}-.*", file_name):
                os.remove(f"../../config/cls_contexts/{file_name}")
                break
        # Remove the cluster from the cluster parameters
        if cls in clus_params:
            del clus_params[cls]

        # Remove the cluster nodes IPs 
        del_cluster_ips(cls)
    # Save the updated cluster parameters file
    with open('../../config/clus_params.yaml', 'w') as file:
        yaml.dump(clus_params, file)

def delete_all_clusters():
    """
    Deletes all clusters specified in the cluster parameters file.

    Args:
    - None
    """
    with open('../../config/clus_params.yaml', 'r') as f:
        clus_params = yaml.safe_load(f)
        clusters_to_delete = list(clus_params.keys())
        del_clusters(clusters_to_delete)


def configure_mirror_registry(cluster_name):
    """
    Configures the Orange mirror registry on all nodes of the specified cluster.

    Args:
        - cluster_name (str): The name of the cluster to configure the Orange mirror registry on.
    """
    nodes = [
        f"{cluster_name}-control-plane",
        f"{cluster_name}-worker"
    ]
    for node in nodes:
        #For each node of the cluster 
        print(node)
        subprocess.run(['docker', 'cp', '../cls_dockerrepo/node_repo_config.sh', f'{node}:/node_repo_config.sh'])
        subprocess.run(['docker', 'cp', '../../config/cni_plugins/.', f'{node}:/opt/cni/bin/'])
        subprocess.run(['docker', 'exec', node, '/bin/bash', '/node_repo_config.sh'])


if __name__ == '__main__':
    if len(sys.argv) > 2:
        # If the first argument is 'create', create the specified number of clusters
        if sys.argv[1] == 'create':
            n_clusters = int(sys.argv[2])
            #create_clusters(n_clusters)
            add_clusters(n_clusters, operation= "new")
        # If the first argument is 'add', add the specified number of clusters
        elif sys.argv[1] == 'add':
            n_clusters = int(sys.argv[2])
            add_clusters(n_clusters, operation= "add")
        # If the first argument is 'delete', delete the specified clusters
        elif sys.argv[1] == 'delete':
            clusters_to_delete = sys.argv[2:]
            del_clusters(clusters_to_delete)
    # If the first argument is 'delete_all', delete all clusters specified in the cluster parameters file
    elif sys.argv[1] == 'delete_all':
            delete_all_clusters()