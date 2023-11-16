import requests
import yaml 
import os 
import time 


def get_list_clusters(exclude_broker = True):   
    """
    Retrieves the list of clusters from the clus_params.yaml file.

    Args:
    - exclude_broker (bool, optional): If True, excludes the "c0" cluster from the list. Default is True.

    Returns:
    - clusters (list): The list of clusters.
    """
    clus_params_file = "/code/clus_params.yaml"
    with open(clus_params_file, "r") as fichier:
        clusters = list(yaml.safe_load(fichier).keys())
    if exclude_broker : 
        clusters.remove("c0")
    return clusters 

def get_pods_resources(thanos, clusters, namespaces, sum_namespaces=True, resources="consumption", cpu_rate="5m"):
    """
        Retreives the resouce informations for pods aggregated by clusters or by requests (namespaces) 

    Args:
        - thanos (str): The URL of the Thanos API.
        - clusters (list): The list of clusters to consider.
        - namespaces (list): The list of namespaces to consider.
        - sum_namespaces (bool, optional): If True, sums the resource consumption per cluster. Default is True.
        - resources (str, optional): The mode of resource information to retrieve. Should be one of the following:
            - "consumption": Retrieves actual consumption data for CPU and RAM using Thanos Prometheus.
            - "limit": Retrieves resource limit data defined in the query slice using Thanos Prometheus.
        - cpu_rate (str, optional): The CPU rate for consumption mode. Default is set to "5m",
            indicating that the "rate" function is applied to a CPU time counter to measure the average CPU usage rate
            in terms of CPU work seconds per second over a 5-minute period.

    Returns:
    - result_dict (dict): The dictionary containing RAM and CPU resources aggregated by clusters or by requests (namespaces).
    """
    result_dict = {}
    for cls in clusters:
        for ns in namespaces :
            #print("chika")
            if resources == "consumption":
                result_dict = get_pods_resources_per_request_cluster(thanos, ns, cls, result_dict, cpu_rate=cpu_rate, resources="consumption")
            elif resources == "limit":
                result_dict = get_pods_resources_per_request_cluster(thanos, ns, cls, result_dict, resources="limit")
            

    result = {}
    for cluster, data_cluster in result_dict.items():
        if (sum_namespaces) and (cluster not in result) :
            result[cluster] = {'RAM': 0, 'CPU': 0}
        for namespace, data_namespace in data_cluster.items():
            if namespace in namespaces:
                if (not sum_namespaces) and (namespace not in result):
                    result[namespace] = {'RAM': 0, 'CPU': 0}
                for _, values in data_namespace.items():
                    if sum_namespaces:
                        result[cluster]['RAM'] += float(values['RAM'])
                        result[cluster]['CPU'] += float(values['CPU'])
                    else: 
                        result[namespace]['RAM'] += float(values['RAM'])
                        result[namespace]['CPU'] += float(values['CPU'])       
    return result


def get_pods_resources_per_request_cluster(thanos, namespace, cluster, result_dict={}, cpu_rate="5m", resources="consumption"):         
    """
    Retrieves the resource information for pods in the specified namespace and cluster. based on the specified resources mode,
    using Thanos Prometheus.

    Args:
        - thanos (str): The URL of the Thanos API.
        - namespace (str): The namespace to query.
        - cluster (str): The cluster to query.
        - result_dict (dict, optional): The dictionary to store the query results. Default is an empty dictionary.
        - cpu_rate (str, optional): The CPU rate for consumption mode. Default is set to "5m", 
            indicating that the "rate" function is applied to a CPU time counter to measure the average CPU usage rate in terms of CPU work seconds per second over a 5-minute period.
        - resources (str, optional): The mode of resource information to retrieve. Should be one of the following:
            - "consumption": Retrieves actual consumption data for CPU and RAM using Thanos Prometheus.
            - "limit": Retrieves resource limit data defined in the query slice using Thanos Prometheus.

    Returns:
        - result_dict (dict): The updated dictionary containing the resource information for pods in the specified namespace and cluster..
    """
 
    if resources == "consumption":
        # Query to retrieve actual consumption data for RAM
        query_ram = f'container_memory_working_set_bytes{{namespace="{namespace}", cluster="cluster-{cluster}", container !="", container!~"^wait-.*"}}/(1024^2)'
        # Query to retrieve actual consumption data for CPU with specified CPU rate
        query_cpu = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}", cluster="cluster-{cluster}", container !=""}}[{cpu_rate}])'
    elif resources == "limit":
        # Query to retrieve resource limit data for RAM
        query_ram = f'kube_pod_container_resource_limits{{namespace="{namespace}",unit="byte",cluster="cluster-{cluster}"}}/(1024^2)'
        # Query to retrieve resource limit data for CPU
        query_cpu = f'kube_pod_container_resource_limits{{namespace="{namespace}",unit="core",cluster="cluster-{cluster}"}}'
    
    # Perform the HTTP requests to Thanos Prometheus to get the results
    os.environ["no_proxy"] = 'localhost,127.0.0.1'
    http_ram_query = f"{thanos}/api/v1/query?query={query_ram}"
    http_cpu_query = f"{thanos}/api/v1/query?query={query_cpu}"


    ram_results = requests.get(http_ram_query)
    cpu_results = requests.get(http_cpu_query)
    #print(cpu_results)
    cpu_data = cpu_results.json()["data"]["result"]
    ram_data = ram_results.json()["data"]["result"]
    if resources == "consumption":
        while (len(cpu_data) != len(ram_data)):
            print(f"CPU: {len(cpu_data)} RAM: {len(ram_data)}")
            time.sleep(2)
            cpu_results = requests.get(http_cpu_query)
            cpu_data = cpu_results.json()["data"]["result"]
            #print(cpu_data)
            ram_results = requests.get(http_ram_query)
            ram_data = ram_results.json()["data"]["result"]
    # Process the retrieved data and update the result_dict dictionary
    for ram_result, cpu_result in zip(ram_data, cpu_data):
        ram_metric = ram_result["metric"]
        ram_value = ram_result["value"][1] 
        cpu_value = cpu_result["value"][1]  

        cluster = ram_metric["cluster"]
        namespace = ram_metric["namespace"]
        container = ram_metric["container"]

        if cluster not in result_dict:
            result_dict[cluster] = {}

        if namespace not in result_dict[cluster]:
            result_dict[cluster][namespace] = {}

        if container not in result_dict[cluster][namespace]:
            result_dict[cluster][namespace][container] = {}

        result_dict[cluster][namespace][container] = {
            "RAM": ram_value,
            "CPU": cpu_value
        }
    
    return result_dict


def get_clusters_consumption(thanos, clusters, cpu_rate = "5m"): 
    """
    Retrieves the CPU and RAM consumption for the worker node in each input cluster using Thanos Prometheus.

    Args:
        - thanos (str): The URL of the Thanos API.
        - clusters (list): A list of cluster names to query.
        - cpu_rate (str, optional): The CPU rate for consumption mode. Default is set to "5m",
            indicating that the "rate" function is applied to a CPU time counter to measure the average CPU usage rate in terms of CPU work seconds per second over a 5-minute period.

    Returns:
        - result_dict (dict): A dictionary containing the CPU and RAM consumption for the worker node of each input cluster.
    """

    # Initialize the dictionary to store the results
    result_dict = {}

    # Queries to retrieve CPU and RAM consumption data for worker nodes
    query_cpu = f'rate(container_cpu_usage_seconds_total{{id="/", node=~".*worker.*"}}[{cpu_rate}])'
    query_ram = f'container_memory_working_set_bytes{{id="/", node=~".*worker.*"}}/(1024^2)'
 
    # Perform the HTTP requests to Thanos Prometheus to get the results
    ram_results = requests.get(f"{thanos}/api/v1/query?query={query_ram}")
    cpu_results = requests.get(f"{thanos}/api/v1/query?query={query_cpu}")

    # Extract the data from the HTTP response
    
    ram_data = ram_results.json()["data"]["result"]
    cpu_data = cpu_results.json()["data"]["result"]

    while len(cpu_data) != len(clusters):
        time.sleep(0.5)
        print(f"CPU: {len(cpu_data)} Clusters: {len(clusters)}")
        cpu_results = requests.get(f"{thanos}/api/v1/query?query={query_cpu}")
        cpu_data = cpu_results.json()["data"]["result"]
    # Process the retrieved data and update the result_dict dictionary
    for ram_result, cpu_result in zip(ram_data, cpu_data):
        cluster = ram_result['metric']['cluster']
        if cluster.split("-")[1] in clusters:  # Check if the cluster is in the input list of clusters
            result_dict[cluster] = {
                'RAM': float(ram_result['value'][1]),  
                'CPU': float(cpu_result['value'][1])   
            }

    return result_dict

def get_clusters_loaded_consumption(thanos_url, clusters, namespaces, cpu_rate = '5m'):
    """
    Calculates the loaded consumption of clusters by subtracting the actual consumption of pods from the max consumption
    and adding the cluster consumption.

    Args:
        - thanos_url (str): The URL of the Thanos API.
        - clusters (list): A list of clusters to query.
        - namespaces (list): A list of namespaces to query.
        - cpu_rate (str, optional): The CPU rate for consumption mode. Default is set to "5m",
            indicating that the "rate" function is applied to a CPU time counter to measure the average CPU usage rate in terms of CPU work seconds per second over a 5-minute period.

    Returns:
        - loaded_consumption (dict): A dictionary containing the loaded consumption of RAM and CPU for each cluster, it is calculated as follows:
            loaded_consumption = max_consumption - actual_consumption + cluster_consumption
    """
    loaded_consumption = {}
    sum_ns = True
    actual_consumption = get_pods_resources(thanos_url, clusters, namespaces, sum_namespaces = sum_ns, resources="consumption", cpu_rate=cpu_rate)
    print("actual", flush= True)
    print(actual_consumption, flush= True)
    max_consumption = get_pods_resources(thanos_url, clusters, namespaces, sum_namespaces = sum_ns, resources="limit")
    print("max", flush= True)
    print(max_consumption, flush= True)
    cluster_consumption = get_clusters_consumption(thanos_url, clusters, cpu_rate = cpu_rate)
    print("consumot", flush= True)
    print(cluster_consumption, flush= True)
    for cluster, consumption in actual_consumption.items():
        loaded_consumption[cluster] = {
            "RAM": max_consumption[cluster]["RAM"] - consumption["RAM"] + cluster_consumption[cluster]["RAM"],
            "CPU": max_consumption[cluster]["CPU"] - consumption["CPU"] + cluster_consumption[cluster]["CPU"],
        }

    for cluster in cluster_consumption:
        if cluster in actual_consumption :
            loaded_consumption[cluster] = {
                "RAM": max_consumption[cluster]["RAM"] - actual_consumption[cluster]["RAM"] + cluster_consumption[cluster]["RAM"],
                "CPU": max_consumption[cluster]["CPU"] - actual_consumption[cluster]["CPU"] + cluster_consumption[cluster]["CPU"],
            }
        else :
            loaded_consumption[cluster] = {
                "RAM":  cluster_consumption[cluster]["RAM"],
                "CPU":  cluster_consumption[cluster]["CPU"],
            }
    return loaded_consumption



def get_all_namespaces():
    pass 
if __name__ == '__main__':

        clusters = get_list_clusters()#["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]
        """ 
        namespaces = []
        result_dict = {}
        print(clusters)
        thanos_url = "http://c0-worker:31885"
        sum_ns = True
        result = get_pods_resources(thanos_url, clusters, namespaces, sum_namespaces = sum_ns, resources="consumption", cpu_rate="5m")
        print(f"NFs consumption: {result}")
        result = get_pods_resources(thanos_url, clusters, namespaces, sum_namespaces = sum_ns, resources="limit")
        print(f"NFs limits: {result}")
        cluster_consumption = get_clusters_consumption(thanos_url, clusters, cpu_rate = "5m")
        print(f"Clusters consumption: {cluster_consumption}")
        result_infra = get_clusters_loaded_consumption(thanos_url, clusters, namespaces, cpu_rate="5m")
        print(f"Loaded clusters consumption: {result_infra}")
        """
        CONNECTION_STRING = "mongodb://mongodb:27017"
        thanos_url = "http://c0-worker:32299"
        namespaces = []#get_all_slices(CONNECTION_STRING)
        clusters = get_list_clusters()
        print(namespaces, flush= True)
        print(clusters, flush= True)
        load = get_clusters_loaded_consumption(thanos_url, clusters, namespaces, cpu_rate="2m")
        print(load, flush= True)