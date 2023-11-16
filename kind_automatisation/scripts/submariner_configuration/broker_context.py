import json 
import yaml
import subprocess


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
    # Extract cluster context 
    subprocess.run(f"docker exec {cluster_name}-control-plane cat /etc/kubernetes/admin.conf > ../../config/cls_contexts/{cluster_name}-control-plane.yaml", shell=True)
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

def install_submariner(broker_name: str, broker_config: str):
    """
    Installs a submariner in the broker cluster name with the given configuration file.

    Args:
        - broker_name (str): The name of the broker to install.
        - broker_config (str): The path to the broker configuration file.
    """
    subprocess.run(['docker', 'cp', broker_config, f'{broker_name}-control-plane:/broker_config.sh'])
    subprocess.run(['docker', 'exec', f'{broker_name}-control-plane', '/bin/bash', '/broker_config.sh'])
    subprocess.run(["kubectl", "wait", "--for=condition=Ready", "--timeout=600s", "pod", "-A", "--all", "--context",f"kind-{broker_name}"], check=True)
def build_broker_context(broker_cluster: str):
    """
    Builds the context file for the broker cluster.

    Args:
        - broker_cluster (str): The name of the broker cluster
    """
    with open("../../config/clus_ips.json") as f:
        clus_ips = json.load(f)

    with open("../../config/clus_params.yaml") as f:
        clus_param = yaml.safe_load(f)

    path = f"../../config/cls_contexts/{broker_cluster}-control-plane.yaml"
    with open(path) as f:
        broker_config = yaml.safe_load(f)

    for key in clus_param:
        if key != broker_cluster:
            path = f"../../config/cls_contexts/{key}-control-plane.yaml"
            with open(path) as f:
                ctx_key = yaml.safe_load(f)
            new_cluster = {
                "cluster": {
                    "certificate-authority-data": ctx_key["clusters"][0]["cluster"]["certificate-authority-data"],
                    "server": f"https://{clus_ips[key]['control-plane'].split('/')[0]}:6443"
                },
                "name": key
            }

            new_context = {
                "context": {
                    'cluster': key,
                    'user': key
                },
                'name': key
            }

            new_user = {
                'name': key,
                'user': {
                    'client-certificate-data': ctx_key["users"][0]["user"]["client-certificate-data"],
                    'client-key-data': ctx_key["users"][0]["user"]["client-key-data"]
                }
            }

            broker_config["clusters"].append(new_cluster)
            broker_config["contexts"].append(new_context)
            broker_config["users"].append(new_user)

    with open(f'../../config/new_broker_config.yaml', 'w') as f:
        yaml.safe_dump(broker_config, f)

def join_broker(broker_name: str, clusters=None, deploy=True):
    """
    Generate and execute a bash script to join the specified broker to the specified clusters.

    Args:
        broker_name (str): Name of the broker to join.
        clusters (Optional[List[str]]): List of cluster names to join. If None, all clusters except the broker's own will be joined.
        deploy (bool): Whether to deploy the broker or only join the deployed one.

    Returns:
        None
    """
    # Load cluster IPs from file
    with open("../../config/clus_ips.json") as f:
        clus_ips = json.load(f)

    # Build bash script
    commandes = [ '#!/bin/bash', "", "export PATH=$PATH:~/.local/bin"]
    if deploy : 
        clusters = clus_ips.keys()
        key = broker_name
        commandes.append(f"kubectl config set-cluster {key} --server https://{clus_ips[key]['control-plane'].split('/')[0]}:6443")
        commandes.append(f"subctl deploy-broker")
        commandes.append(f"kubectl annotate node {key}-worker gateway.submariner.io/public-ip=ipv4:{clus_ips[key]['worker'].split('/')[0]}")
        commandes.append(f"kubectl label node {key}-worker submariner.io/gateway=true")
        commandes.append(f"subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid kind-{key}")
    for key in clusters:
        # For each cluster to join, add kubectl and subctl commands to the bash script
        if key != broker_name:
            # Joining the broker's own cluster requires deploying the broker using subctl
            commandes.append(f"kubectl annotate node {key}-worker gateway.submariner.io/public-ip=ipv4:{clus_ips[key]['worker'].split('/')[0]} --context {key}")
            commandes.append(f"kubectl label node {key}-worker submariner.io/gateway=true --context {key}")
            commandes.append(f"subctl join broker-info.subm --natt=false --force-udp-encaps --clusterid {key} --context {key}")


    # Write bash script to file
    commandes_str = '\n'.join(commandes)
    with open("./broker_join.sh", "w+") as f:
        f.write(commandes_str)

    subprocess.run(f"docker cp ../../config/new_broker_config.yaml {broker_name}-control-plane:/etc/kubernetes/admin.conf", shell=True, check=True)
    subprocess.run(f"docker cp ./broker_join.sh {broker_name}-control-plane:/broker_join.sh", shell=True, check=True)
    subprocess.run(f"docker exec {broker_name}-control-plane chmod +x /broker_join.sh", shell=True, check=True)
    subprocess.run(f"docker exec {broker_name}-control-plane /broker_join.sh", shell=True, check=True)


if __name__ == '__main__':
    pass