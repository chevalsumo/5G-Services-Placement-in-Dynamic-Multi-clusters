import argparse
import yaml 
import subprocess
import json
from pymongo import MongoClient

def save_cluster_limits(cluster_params, method = "DB", mongoDBURL = ""):
    """
    Save the resource limits applied to the various clusters 
    either in a json file or directly on mongoDB.

    Args:
    - cluster_params (dict): Dictionary containing cluster parameters (RAM and CPU limits).
    - method (str): Method of saving the cluster limits. Options: "File" or "DB". Defaults to "DB".
    - mongoDBURL (str): The URL of the MongoDB server. Required if method is set to "DB".

    """
    if method == "File":
        with open('../../config/clus_resources.json', 'r') as file:
            saved_cluster_params = json.load(file)
        for cls in cluster_params : 
            saved_cluster_params[cls] = cluster_params[cls]
        with open('../../config/clus_resources.json', 'w') as file:
            json.dump(saved_cluster_params, file)
    elif method == "DB":
        client = MongoClient(mongoDBURL)
        infra = client["NSlies"]['infra']
        for cls in cluster_params :
            document = {
                "ID" :  f'cluster-{cls}',
                "ram" : cluster_params[cls]["ram"],
                "cpu" : cluster_params[cls]["cpu"]
            }
            cluster = infra.find_one({"ID": cls})
            if cluster :
                infra.update_one({"ID": cls}, {"$set": {"ram": document["ram"], "cpu": document["cpu"]}})
            else :
                infra.insert_one(document)
def limite_cluster_resources(cluster_params):
    """
    Limits the CPU and RAM resources of a clusters workers.

    Args:
    - cluster_params (dict): A dictionary containing cluster parameters.
                            Each key is the cluster name, and the value is another dictionary
                            with 'cpu' and 'ram' values specifying the resource limits.
    """
    for cls in cluster_params: 
        # CPU constraint
        subprocess.run(['docker', "update","--cpus", f"{cluster_params[cls]['cpu']}",f"{cls}-worker"])
        # RAM constraint 
        subprocess.run(['docker', "update","--memory", f"{cluster_params[cls]['ram']}M","--memory-swap",  f"{cluster_params[cls]['ram'] + 1}M", f"{cls}-worker"])

if __name__ == '__main__':  
    """
    If you want to specify for each cluster use the -c option followed 
    by the cluster name, the CPU limit (number of cores) and the RAM limit in MB.

    Use --all-cpu and --all-ram for all other unspecified clusters 

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--cluster', action='append', metavar=('name', 'cpu', 'ram'), nargs=3, help='CPU and RAM limits for a specific cluster')
    parser.add_argument('--all-cpu', type=int, help='CPU limit for all remain clusters')
    parser.add_argument('--all-ram', type=str, help='RAM limit for all remain clusters')
    args = parser.parse_args()
    with open('../../config/clus_params.yaml', 'r') as file:
        all_clusters_keys = list(yaml.safe_load(file).keys())
        
    cluster_params = {}
    if args.cluster:
        for cluster_data in args.cluster:
            cluster_params[f"{cluster_data[0]}"] = {'cpu': int(cluster_data[1]), 'ram': int(cluster_data[2])}
    if args.all_cpu and args.all_ram:
        for cls in all_clusters_keys:
            if cls not in cluster_params:
                cluster_params[f"{cls}"] = {'cpu': int(args.all_cpu), 'ram': int(args.all_ram)}

    save_cluster_limits(cluster_params, method = "DB", mongoDBURL = "mongodb://localhost:27017")
    limite_cluster_resources(cluster_params) 


    #Usage exemple python cluster_resources.py --all-ram 6144 --all-cpu 3