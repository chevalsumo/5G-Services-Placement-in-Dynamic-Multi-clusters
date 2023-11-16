import sys
import os
import yaml
from BD_management import *
import requests
import numpy as np
import subprocess
import time
from api import *
from sklearn.cluster import SpectralClustering
import random
from clustering_tools import *

class AugmentedGraph:

    def __init__(self, nb_cluster, _clusters = None):
        self._nb_cluster = nb_cluster 
        self._clusters = _clusters

    def get_clusters(self, _adj_attrs = None, nb_cluster = None):
        if nb_cluster is None :
            nb_cluster = self._nb_cluster
        sc = SpectralClustering(n_clusters=nb_cluster, affinity= "precomputed")
        mod = sc.fit_predict(_adj_attrs)
        nb_nodes = _adj_attrs.shape[0]
        clusters = {
            i : [] for i in range(nb_cluster)
        }
        for i in range(nb_nodes) :
            clusters[mod[i]].append(i)
        return clusters
        
    def get_final_action(self, _selected_cluster, MAX_DEFAULT_NODE_DELAY, _vnf_obs = None, choice = "Min_DELAY"):
        nodes = self._clusters[_selected_cluster]
        _selected = self.select_nod_cluster(nodes, self._costs, self._pns_attrs, MAX_DEFAULT_NODE_DELAY, choice, _vnf_obs = _vnf_obs)
        return _selected
    
    def get_augmented_graph(self, _adj_attrs = None, _pns_attrs = None, keep_clusters = True, _costs = False, MAX_DEFAULT_NODE_DELAY = False, choice = "Max_RAM"):


        #if  self._pns_attrs.shape[0] !=  _pns_attrs.shape[0]:
            # Infrastrcture change 
            # Hint use prometheus keys to track which cluster deleted or added
        #    pass
        self._pns_attrs = _pns_attrs
        self._adj_attrs = _adj_attrs
        self._costs = _costs

        
        if (not keep_clusters) or (self._clusters is None):
            self._clusters = self.get_clusters(_adj_attrs = _adj_attrs)
        _cls_costs = []
        _cls_attrs = None
        for _id, cls in self._clusters.items():
            _selected = self.select_nod_cluster(cls,_costs, _pns_attrs, MAX_DEFAULT_NODE_DELAY, choice)
            _cl_attrs = _pns_attrs[_selected, :]
            if _cls_attrs is None :
                    _cls_attrs = _cl_attrs
            else :
                    _cls_attrs = np.vstack((_cls_attrs, _cl_attrs))

            
            _cls_costs.append(_costs[_selected])
        _cls_costs = np.array(_cls_costs)

        return _cls_attrs, _cls_costs
    
    def select_nod_cluster(self, cls, _costs, _pns_attrs, MAX_DEFAULT_NODE_DELAY, choice, _vnf_obs = None):
        _eligibles = [cls[i] for i in np.where(_costs[cls] < MAX_DEFAULT_NODE_DELAY)[0]]
        if _vnf_obs :
            final = []
            for j in _eligibles :
                if (_vnf_obs[0] <= _pns_attrs[j, 0]) and  (_vnf_obs[1] <= _pns_attrs[j, 1]):
                    final.append(j)
            _eligibles = final 

        if _eligibles: 
            if choice == "Max_RAM":
                # We will take RAM because it's the most critical resource
                _attrs_eligibles = _pns_attrs[_eligibles, :]
                _selected = _eligibles[np.argmax(_attrs_eligibles[: ,1])]
            elif choice == "Min_DELAY":
                # Choose the nearest node as cluster representative 
                _selected = _eligibles[np.argmin(_costs[_eligibles])]

        else:
            _selected = random.choice(cls)
        return _selected
    
def adapte_mutus_subnets(CHART, values, subnet, n6_subnet = None):
    """
    Modifies the subnets of N2/N3/N6 networks used in network function charts (AMF/SMF/GNB/UPF) deployments for each slice.

    Args:
    - CHART (str): The type of network function chart ("amf", "smf", "upf", "gnb").
    - values (dict): The dictionary containing the network function values.
    - subnet (int): The subnet value to use for (N2/N3/N4) networks.
    - n6_subnet (int, optional): The subnet value to use for N6 network. Default is None.

    Returns:
    - values (dict): The modified dictionary containing the updated network function values.
    """

    if CHART == "amf":
        # Modify the values for AMF chart
        if "global" not in values:
            values["global"] = {}

        if "amf" not in values["global"]:
            values["global"]["amf"] = {}

        if "n2if" not in values["global"]["amf"]:
            values["global"]["amf"]["n2if"] = {}

        if "n2network" not in values["global"]:
            values["global"]["n2network"] = {}

        values["global"]["amf"]["n2if"]["ipAddress"] = f"10.{subnet}.249"
        values["global"]["n2network"]["subnetIP"] = f"10.{subnet}.248"
        values["global"]["n2network"]["gatewayIP"] = f"10.{subnet}.254"
        values["global"]["n2network"]["excludeIP"] = f"10.{subnet}.254"
    elif CHART == "smf":
        # Modify the values for SMF chart
        if "global" not in values:
            values["global"] = {}
        if "smf" not in values["global"]:
            values["global"]["smf"] = {}
        if "n4if" not in values["global"]["smf"]:
            values["global"]["smf"]["n4if"] = {}
        if "n4network" not in values["global"]:
            values["global"]["n4network"] = {}

        values["global"]["smf"]["n4if"]["ipAddress"] = f"10.{subnet}.244"
        values["global"]["n4network"]["subnetIP"] = f"10.{subnet}.240"
        values["global"]["n4network"]["gatewayIP"] = f"10.{subnet}.246" 
        values["global"]["n4network"]["excludeIP"] = f"10.{subnet}.246" 
    elif CHART == "upf":
        # Modify the values for UPF chart
        if "global" not in values:
            values["global"] = {}
        if "upf" not in values:
            values["upf"] = {}
        if "n4if" not in values["upf"]:
            values["upf"]["n4if"] = {}
        if "n3if" not in values["upf"]:
            values["upf"]["n3if"] = {}
        if "n6if" not in values["upf"]:
            values["upf"]["n6if"] = {}
        if "n3network" not in values["global"]:
            values["global"]["n3network"] = {}
        if "n4network" not in values["global"]:
            values["global"]["n4network"] = {}
        if "n6network" not in values["global"]:
            values["global"]["n6network"] = {}

        values["upf"]["n3if"]["ipAddress"] = f"10.{subnet}.233"
        values["upf"]["n4if"]["ipAddress"] = f"10.{subnet}.241"
        values["upf"]["n6if"]["ipAddress"] = f"10.{n6_subnet}.12"

        values["global"]["n3network"]["subnetIP"] = f"10.{subnet}.232"
        values["global"]["n3network"]["gatewayIP"] = f"10.{subnet}.238"
        values["global"]["n3network"]["excludeIP"] = f"10.{subnet}.238"

        values["global"]["n4network"]["subnetIP"] = f"10.{subnet}.240"
        values["global"]["n4network"]["gatewayIP"] = f"10.{subnet}.246" 
        values["global"]["n4network"]["excludeIP"] = f"10.{subnet}.246"

        values["global"]["n6network"]["subnetIP"] = f"10.{n6_subnet}.0"
        values["global"]["n6network"]["gatewayIP"] = f"10.{n6_subnet}.1" 
        values["global"]["n6network"]["excludeIP"] = f"10.{n6_subnet}.254"
    elif CHART == "gnb":
        # Modify the values for GNB UERANSIM chart
        if "global" not in values:
            values["global"] = {}
        if "gnb" not in values:
            values["gnb"] = {}

        values["global"]["n2network"] = {"subnetIP" : f"10.{subnet}.248",
                                         "gatewayIP" : f"10.{subnet}.254",
                                         "excludeIP" : f"10.{subnet}.254"}

        values["global"]["n3network"] = {"subnetIP" : f"10.{subnet}.232",
                                         "gatewayIP" : f"10.{subnet}.238",
                                         "excludeIP" : f"10.{subnet}.238"}

        values["gnb"]["n2if"] = {"ipAddress" : f"10.{subnet}.250"}
        values["gnb"]["n3if"] = {"ipAddress" : f"10.{subnet}.236"}
        values["gnb"]["amf"] = {"n2if":{"ipAddress" : f"10.{subnet}.249"}}
    return values
def install_NF(CHARTS_PATH, NAMESPACE, context, CHART, ressources, subnet, n6_subnet = None, timeout = "2m0s"):
    """
    Install NF chart in a the namespace of a specefic cluster.

    Args:
    - CHARTS_PATH (str): The path to the charts directory.
    - NAMESPACE (str): The namespace to deploy the chart.
    - subnet (int): The subnet value to use for (N2/N3/N4) networks.
    - n6_subnet (int, optional): The subnet value to use for N6 network. Default is None.
    - context (str): The context to use for the Kubernetes cluster.
    
    - CHART (str): The name of the chart to install.
    """
    # Create custom values to use in the NFs charts to use the exported services (nrf/mongoDB) by submariner
    values = {
        "global": {
            "nrf": {
                "service": {
                    "name": f"nrf-nnrf.{NAMESPACE}.svc.clusterset.local"
                }
            }
        },
        "mongodb": {
            "service": {
                "name": f"mongodb.{NAMESPACE}.svc.clusterset.local"
            }
        },
        
    }
    
    


    if CHART != "nrf":
        values = {
            "global": {
                "nrf": {
                    "service": {
                        "name": f"nrf-nnrf.{NAMESPACE}.svc.clusterset.local"
                    }
                }
            },
            "mongodb": {
                "service": {
                    "name": f"mongodb.{NAMESPACE}.svc.clusterset.local"
                }
            },
           CHART : {
               "resources" : ressources
           }
        }       
    else : 
        values = {
            "nrf" : {
                "resources" : ressources["nrf"]
            },
            "mongodb" : {
                "resources" : ressources["mongodb"]  
            }
       }

    values = adapte_mutus_subnets(CHART, values, subnet, n6_subnet = n6_subnet)
    VALUES_PATH = "./nfs_values.yaml"
    with open(VALUES_PATH, "w") as f:
        yaml.dump(values, f)
    values_file = f"-f {VALUES_PATH}"
    
    helm_command = f"helm install {CHART.lower()} {CHARTS_PATH} -n {NAMESPACE} --kube-context {context} {values_file} --create-namespace --wait --timeout {timeout}"
    print(helm_command)
    #os.system(helm_command)
    process = subprocess.Popen(helm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print("Helm chart installed successfully. Deployed containers are running")
    else:
        print(f"Error installing Helm chart: {stderr.decode()}")
        return False

def uninstall_NF(NAMESPACE, context, CHART):
    """
    Uninstalls a chart.

    Args:
    - NAMESPACE (str): The namespace where the chart is deployed.
    - context (str): The context to use for the Kubernetes cluster.
    - CHART (str): The name of the chart to uninstall.
    """
    helm_command = f"helm -n {NAMESPACE} --kube-context {context} uninstall {CHART}"
    os.system(helm_command)

def uninstall_placed_NFs(NAMESPACE, NFs_placement):

    for nf in NFs_placement:
        uninstall_NF(NAMESPACE, NFs_placement[nf], nf)
if __name__ == "__main__":
    
    # Number of clusters used in the training (number of RL actions )
    NB_clusters = 4

    # The politic of clustering (keep clusters or change each step)
    keep_clusters = True 

    # Get the trained clusters 
    #_clusters = {0: [3, 6, 7], 1: [4, 5], 2: [0, 1, 2], 3: [8, 9, 10]}
    _result_clus = requests.get(f'http://mlflow_ms_api:1234/base_clusters').json()["clusters"]
    _result_clus = {int(key): value for key, value in _result_clus.items()}
    print(_result_clus)
    AG = AugmentedGraph(NB_clusters, _clusters = _result_clus) 
    
    # Obtain the graph's topology and attributes.
    adj_matrix, nsparse_adj_list, _adj_attrs, edges_clusters = define_topology()
    print(edges_clusters)

    # Create the domain for our topology 

    MAX_DEFAULT_NODE_DELAY = 20.0 #The Fixed max delay used on training

    _domain = Domain(0, "0", adj_matrix, nsparse_adj_list, None, None, _adj_attrs = _adj_attrs,
    _deployed = True)


    # Get the current value of the PATH environment variable
    path = os.environ.get('PATH')

    # Check if ~/.local/bin is already in the PATH
    if '~/.local/bin' not in path:
        # Append ~/.local/bin to the PATH
        path += ':~/.local/bin'

        # Set the new value of the PATH environment variable
        os.environ['PATH'] = path
    NAMESPACE = "salimou"
    TYPE = "v2x"
    CONNECTION_STRING = "mongodb://mongodb:27017"
    response = requests.get(f'http://requests_gen:8000/rg/{NAMESPACE}/{TYPE}').json()
    #print(response["sr"])
    doc, sr, delay = response["sr"], response["doc"], response["delay"]
    _VNFs = {}
    #print(sr)
    #add_nfs_slice(CONNECTION_STRING, NAMESPACE, doc)
    
    print(delay)
    nfs_placement = {}
    nf_index = 0
    nfs = list(sr["nfs"].keys())

    _order = ["ue", "nrf" ,"amf", "gnb", "upf", "DN"]
    _sorted_keys  = _order +  [key for key in nfs if key not in _order]
    placement_fail = False
    i, j = get_offset_subnets(CONNECTION_STRING)
    print(f"I:        {i}      J:            {j}")
    if i is None :
        i, j = 0,0 
    else :
        if i == 49: 
            j += 0
            i = 0
        else:
            i+= 1

        #j+= 1 

    #set_offset_subnets(CONNECTION_STRING, i, j)
    add_slice_req(CONNECTION_STRING, NAMESPACE, TYPE)

    # Last Kubernetes cluster used where the precedent NF is placed
    _last_cluster = None 

    while nf_index < len(_sorted_keys) and not placement_fail:
        nf = _sorted_keys[nf_index]
        print(f"{nf}")
        _cpu = sr["nfs"][nf]['cpu_limits'] / 1000
        _ram = sr["nfs"][nf]['ram_limits']
        if nf != "ue":
            _required_delay = delay[nf]["delay"]
            _VNFs[nf] = [_cpu, _ram, _required_delay]
        else:
             _VNFs[nf] = [_cpu, _ram, MAX_DEFAULT_NODE_DELAY]
        print("Get cluster resources")
        limits_cls = read_root()['infra_db']
        print("Get Infra limits")
        loaded_cls = clusters_all_loaded()['loaded_clusters']
        _infra_attributes = None
        dispo = get_available_resources(limits_cls, loaded_cls)
        print("Available resources")
        print(dispo)
        infra_data = []
        for cluster, resources in dispo.items():
            infra_data.append([resources['CPU'], resources['RAM']])

        _infra_attributes = infra_data #np.array(infra_data)
        if nf == "ue":
            # The UE NF is placed in randmly at an edge cluster
            _ue_cluster = random.choice(edges_clusters[0])
            cluster = _ue_cluster +1

            _last_cluster = _ue_cluster
            _selected = _last_cluster
        else:
            # If it's not the UE, the NF is placed using our Clustering + RL solution

            # Start by calculating the shortest paths to all the other nodes starting from the last node chosen 

            _selected = int(nfs_placement[delay[nf]["source"]].split("c")[1])-1
            print(f"Delay from the source {delay[nf]['source']} placed in: {_selected}")
            _costs = _domain.dijkstra(_selected, mask_delay = True, 
            MAX_DEFAULT_NODE_DELAY = MAX_DEFAULT_NODE_DELAY,
            required_delay =_required_delay,
            )[0]

            _cls_attrs, _cls_costs = AG.get_augmented_graph(_adj_attrs = _adj_attrs[:,:,1], _pns_attrs = np.array(_infra_attributes), _costs= _costs, MAX_DEFAULT_NODE_DELAY=  MAX_DEFAULT_NODE_DELAY, choice = "Min_DELAY", keep_clusters= True)
            
            obs = {
                "Infra" : np.hstack((_cls_attrs,  _cls_costs.reshape(-1, 1))).tolist(),
                "Vnf" : _VNFs[nf]
            }

            data = {
                "input": obs
            }

            url = "http://mlflow_ms_api:1234/predict"
            response = requests.post(url, json=data)
            print(data)
            if response.status_code == 200:
                result = response.json()
                print("Cluster Prediction:", result["prediction"])
            else:
                print("Error:", response.text)

            _selected = AG.get_final_action(result['prediction'], MAX_DEFAULT_NODE_DELAY= MAX_DEFAULT_NODE_DELAY, choice = "Max_RAM", _vnf_obs = obs["Vnf"])

            
            print(f"Final prediction: {_selected}")
            cluster = _selected +1
    
        context = f"c{cluster}"
        subnet, n6_subnet = f"{100+j}.{50+i}",f"{100+j}.{100+i}"
        if (dispo[f"cluster-{context}"]["RAM"] < _ram) or (dispo[f"cluster-{context}"]["CPU"] < _cpu) :
            print(f"not enough resources in the selected cluster ({context})")
            placement_fail = True
        else :
            if nf != "ue":
                if _costs[_selected] == MAX_DEFAULT_NODE_DELAY:
                    print(f"Delay Problem")
                    placement_fail = True

        if not placement_fail :           
            print("Succ")
            #_last_cluster = _selected
            if True :
                install_NF(doc[nf]["default_values"], NAMESPACE, context, nf, doc[nf]["resources"], subnet, n6_subnet= n6_subnet)
                if nf == "nrf" :
                    mongo_db_export_command = f"~/.local/bin/subctl export service -n {NAMESPACE} mongodb --context {context}"
                    nrf_export_command = f"~/.local/bin/subctl export service -n {NAMESPACE} nrf-nnrf --context {context}"
                    os.system(mongo_db_export_command)
                    os.system(nrf_export_command)
                nfs_placement[nf] = context

        nf_index += 1 

    if True :
        # Uninstall deployed NFs charts if failed 
        if placement_fail:
            print("Placement Fail")
            uninstall_placed_NFs(NAMESPACE, nfs_placement)
            remove_slice_request(CONNECTION_STRING, NAMESPACE)
        else :
            add_nfs_slice(CONNECTION_STRING, NAMESPACE, doc)

"""     
    CHARTS_PATH = "../../towards5gs-helm/charts"  
    # Chart of a Slice Request that we will deploy :
    SR = {
        "nrf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-nrf",
            "resources": {
                "nrf": {
                    "requests": {
                        "cpu": "40m",
                        "memory": "64Mi"
                    },

                    "limits": {
                        "cpu": "50m",
                        "memory": "128Mi"
                    }
                },
                "mongodb": {
                    "requests": {
                        "cpu": " 70m",
                        "memory": "200Mi"
                    },
                    "limits": {
                        "cpu": "100m",
                        "memory": "256Mi"
                    }
                }
            }
        },
        "udr": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-udr",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "udm": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-udm",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "ausf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-ausf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "nssf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-nssf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "amf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-amf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "pcf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-pcf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "smf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-smf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "webui": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-webui",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "upf": {
            "default_values": f"{CHARTS_PATH}/free5gc/charts/free5gc-upf",
            "resources": {
                "requests": {
                    "cpu": "40m",
                    "memory": "64Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "128Mi"
                }
            }
        },
        "ue": {
            "default_values": f"{CHARTS_PATH}/ueransim/charts/ue",
            "resources": {
                "requests": {
                    "cpu": "50m",
                    "memory": "128Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "200Mi"
                }
            }
        },
        "gnb": {
            "default_values": f"{CHARTS_PATH}/ueransim/charts/gnb",
            "resources": {
                "requests": {
                    "cpu": "50m",
                    "memory": "256Mi"
                },
                "limits": {
                    "cpu": "60m",
                    "memory": "300Mi"
                }
            }
        },
    }

    NFs_PLACEMENT = {
        "nrf"  : "kind-c1",
        "udr"  : "kind-c1",
        "udm"  : "kind-c1",
        "ausf" : "kind-c1",
        "nssf" : "kind-c1",
        "amf"  : "kind-c1",
        "pcf"  : "kind-c2",
        "smf"  : "kind-c2",
        "webui": "kind-c2",
        "upf"  : "kind-c2",
        "ue"   : "kind-c2",
        "gnb"  : "kind-c2"
        }
    #NFs_PLACEMENT = { "gnb"  : "kind-c1",} 
    # "gnb"  : "kind-c1"}
    NAMESPACE = sys.argv[1]
    OPERATION = sys.argv[2]
    i = 3
    subnet, n6_subnet = f"100.{50+i}",f"100.{100+i}"
    for chart, context in NFs_PLACEMENT.items():
        if OPERATION == "uninstall":
            helm_command = f"helm -n {NAMESPACE} --kube-context {context} uninstall {chart}"
            os.system(helm_command)
        elif OPERATION == "install":
            install_NF(SR[chart]["default_values"], NAMESPACE, context, chart, SR[chart]["resources"], subnet, n6_subnet= n6_subnet)


    # Exporter les services MongoDB et NrF sur submariner 
    CONNECTION_STRING = "mongodb://localhost:27017"

    if OPERATION == "install" :
        mongo_db_export_command = f"subctl export service -n {NAMESPACE} mongodb --context {NFs_PLACEMENT['nrf']}"
        nrf_export_command = f"subctl export service -n {NAMESPACE} nrf-nnrf --context {NFs_PLACEMENT['nrf']}"
        os.system(mongo_db_export_command)
        os.system(nrf_export_command)
        add_slice_req(CONNECTION_STRING, NAMESPACE, "CN/RN")
        add_nfs_slice(CONNECTION_STRING, NAMESPACE, SR)
    elif OPERATION == "uninstall":
        remove_slice_request(CONNECTION_STRING, NAMESPACE)
        remove_nfs_slice(CONNECTION_STRING, NAMESPACE)
"""
