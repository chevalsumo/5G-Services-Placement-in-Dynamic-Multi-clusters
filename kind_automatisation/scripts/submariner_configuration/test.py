import subprocess
import json
import re
import yaml

def add_cluster_ips(cluster_name, save=True):
    """
    Adds the IPs for the specified cluster.

    Args:
        cluster_name (str): The name of the cluster.
        save (bool, optional): Whether to save the IPs to the file. Defaults to False.

    Returns:
        dict: A dictionary containing the IPs for the specified cluster.
    """
    ips = {}
    ips['control-plane'] = subprocess.check_output(f"docker exec {cluster_name}-control-plane ip a | grep -A 2 'eth0@' | grep -oP 'inet \K[\d./]+'", shell=True, text=True).strip()
    ips['worker'] = subprocess.check_output(f"docker exec {cluster_name}-worker ip a | grep -A 2 'eth0@' | grep -oP 'inet \K[\d./]+'", shell=True, text=True).strip()

    if save:
        with open('../../config/clus_ips.json', 'r+') as file:
            data = json.load(file)
            data[cluster_name] = ips
            file.seek(0)
            json.dump(data, file, indent=4)
    else:
        return ips


def create_cluster_ips_file():
    """
    Creates the cluster IPs file with IPs for all clusters.
    """
    ips = {}
    with open('../../config/clus_params.yaml', 'r') as file:
        cluster_names = yaml.safe_load(file).keys()

    for cluster_name in cluster_names:
        ips[cluster_name] = add_cluster_ips(cluster_name, save=False)

    with open('../../config/clus_ips.json', 'w') as file:
        json.dump(ips, file, indent=4)


def del_cluster_ips(cluster_name):
    """
    Deletes the IP information for the specified cluster.

    Args:
        - cluster_name (str): The name of the cluster to delete the IP information for.
    """
    with open('../../config/clus_ips.json', 'r') as file:
        ips_data = json.load(file)
        ips_data.pop(cluster_name, None)
    
    with open('../../config/clus_ips.json', 'w') as file:
        json.dump(ips_data, file)

if __name__ == '__main__':
    #create_cluster_ips_file()
    #add_cluster_ips("c10", save=True)
    del_cluster_ips("c12")
    pass