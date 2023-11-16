import pandas as pd 
import numpy as np
import sys 
sys.path.append('../../')
from vnf_placement.SubstrateNetwork.SN import *
from vnf_placement.VNFR.VNFR import * 
from vnf_placement.PlacementModule.Mlflow import * 
from vnf_placement.PlacementModule.PlacementEnv import *
from vnf_placement.PlacementModule.PlacementModule import *
#from kind_automatisation.scripts.ModulePlacement.BD_management import *
import os 
import pickle
import random
import requests
import mlflow
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from mlflow.tracking import MlflowClient

def matrix_to_sparse_adjacency(matrix):
    sparse_adjacency = []
    
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            if matrix[i][j] != 0:
                sparse_adjacency.append([i,j])
    
    return sparse_adjacency

def generate_random_resource_configuration(percentage, clusters):
    total_ram = sum(cluster['RAM'] for cluster in clusters.values())
    target_ram = total_ram * (percentage / 100)
    
    available_clusters = list(clusters.keys())
    selected_clusters = []
    
    while target_ram > 0 and available_clusters:
        cluster = random.choice(available_clusters)
        available_ram = clusters[cluster]['RAM']
        
        if available_ram <= target_ram:
            selected_clusters.append(cluster)
            target_ram -= available_ram
        
        available_clusters.remove(cluster)
    
    if target_ram > 0:
        return None  # Impossible to achieve the desired percentage
    
    config = {cluster: clusters[cluster]['RAM'] for cluster in selected_clusters}
    return config

if __name__ == "__main__":


    dispo = {'cluster-c1': {'RAM': 4372.95703125, 'CPU': 2.7409564815426513}, 
            'cluster-c10': {'RAM': 4489.69140625, 'CPU': 2.7610473437294}, 
            'cluster-c2': {'RAM': 4493.51171875, 'CPU': 2.7736120196913383},
            'cluster-c3': {'RAM': 4495.4609375, 'CPU': 2.7832016581886196},
            'cluster-c4': {'RAM': 4461.375, 'CPU': 2.757656933342214}, 
            'cluster-c5': {'RAM': 4501.0390625, 'CPU': 2.767851993789014}, 
            'cluster-c6': {'RAM': 4494.31640625, 'CPU': 2.7732968842827685},
            'cluster-c7': {'RAM': 4498.8125, 'CPU': 2.7635888338288512},
            'cluster-c8': {'RAM': 4468.6484375, 'CPU': 2.7840575406645467},
            'cluster-c9': {'RAM': 4448.27734375, 'CPU': 2.764806127839378}, 
            'cluster-c11': {'RAM': 4512.73828125, 'CPU': 2.7766413269517054}}
    adj_list = [
    [1,2],
    [3],
    [3,4,6],
    [6,7],
    [5],
    [6],
    [7],
    [8,10]
    ,[9]
    ,[10],
      [] ]
    print(adj_list)
    num_nodes = len(adj_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            adj_matrix[node, neighbor] = 1
            adj_matrix[neighbor,node ] = 1

    nsparse_adj_list = matrix_to_sparse_adjacency(adj_matrix)

    DEFAULT_PN = {
    "Edge" : {
        "CPU" : [7],
        "RAM" : [8192],
    },
    "Transport" : {
        "CPU" : [7],
        "RAM" : [8192],
    },
    "Core" : {
        "CPU" : [7],
        "RAM" : [8192],
    }
}

    DEFAULT_PL = {
        "Edge" : {
            "BW" : [50],
            "Delay" : [3],
        },
        "Transport" : {
            "BW" : [50],
            "Delay" : [2.5],
        },
        "Core" : {
            "BW" :  [50],
            "Delay" : [2],
        }
    }

    DEFAULT_VNF = {
        "CPU" : [0.05, 1],
        "RAM" : [64, 1024],
    }

    DEFAULT_VL = {
        "BW" : [2,3,4],
        "Delay" : [2,5]
    }

    #Define boundaries of each variable of our state 
    adj_mat = adj_matrix
    #Boundaries of VNFs, VLs
    RL_boundaries = {}

    RL_boundaries["MAX_DEFAULT_VNFS_CPU"] = max(DEFAULT_VNF["CPU"])
    RL_boundaries["MIN_DEFAULT_VNFS_CPU"] = min(DEFAULT_VNF["CPU"])
    RL_boundaries["MAX_DEFAULT_VNFS_RAM"] = max( DEFAULT_VNF["RAM"] )
    RL_boundaries["MIN_DEFAULT_VNFS_RAM"] = min( DEFAULT_VNF["RAM"])

    RL_boundaries["MAX_SERVICES_BW"] = max(DEFAULT_VL["BW"]) * 2 
    RL_boundaries["MAX_SERVICES_DELAY"] = max(DEFAULT_VL["Delay"])

    RL_boundaries["MIN_SERVICES_BW"] = min(DEFAULT_VL["BW"]) 
    RL_boundaries["MIN_SERVICES_DELAY"] = min(DEFAULT_VL["Delay"])

    #Number of VNFs in a VNFR
    RL_boundaries["MIN_SERVICES_VNFS"], RL_boundaries["MAX_SERVICES_VNFS"] = [8 , 8]



    #Boundaries of PNs, PLs
    #PLs
    RL_boundaries["MAX_DEFAULT_BW"] = max(max(DEFAULT_PL[d]["BW"]) for d in DEFAULT_PL)
    RL_boundaries["MIN_DEFAULT_BW"] = 0


    RL_boundaries["MAX_DEFAULT_DELAY"] = max(max(DEFAULT_PL[d]["Delay"]) for d in DEFAULT_PL)
    RL_boundaries["MIN_DEFAULT_DELAY"] = min(min(DEFAULT_PL[d]["Delay"]) for d in DEFAULT_PL)

    #PNs
    RL_boundaries["MAX_DEG"] = max(int(adj_mat[i,:].sum()) for i in range(adj_mat.shape[0]))
    RL_boundaries["MIN_DEG"] = min(int(adj_mat[i,:].sum()) for i in range(adj_mat.shape[0]))

    RL_boundaries["MAX_DEFAULT_NODE_BW"] = RL_boundaries["MAX_DEFAULT_BW"] * RL_boundaries["MAX_DEG"]
    RL_boundaries["MIN_DEFAULT_NODE_BW"] = 0

    RL_boundaries["MAX_DEFAULT_NODE_DELAY"] = 20.0#RL_boundaries["MAX_DEFAULT_DELAY"]
    RL_boundaries["MIN_DEFAULT_NODE_DELAY"] = 0

    RL_boundaries["MAX_DEFAULT_CPU"] = max(max(DEFAULT_PN[d]["CPU"]) for d in DEFAULT_PN) 
    RL_boundaries["MAX_DEFAULT_RAM"] = max(max(DEFAULT_PN[d]["RAM"]) for d in DEFAULT_PN)   

    RL_boundaries["MIN_DEFAULT_CPU"] = 0 
    RL_boundaries["MIN_DEFAULT_RAM"] = 0  
    
    RL_boundaries["MIN_DEFAULT_NODE_BW"] = 0

    _domain_id = 0
    nb_nodes = adj_mat.shape[0]
    _PNS = {}
    nodes_type = [1, 0, 0, 2, 1, 0, 2, 2, 0, 1, 0]#[1 for i in range(nb_nodes)]
    edges_clusters = []

    edges_clusters = {  0 : [0, 4, 9] 
                        #[i for i in range(nb_nodes)]
                    }
    clusters = list(dispo.keys()) 
    print(edges_clusters)
    for i in range(nb_nodes): 
        if nodes_type[i] == 0 :
            _type = "Transport"
        elif nodes_type[i] == 1 :
            _type = "Edge"
        elif nodes_type[i] == 2 :
            _type = "Core"

        _cpu = dispo[clusters[i]]["CPU"]#random.choice(DEFAULT_PN[_type]["CPU"])
        _ram = dispo[clusters[i]]["RAM"]#random.choice(DEFAULT_PN[_type]["RAM"])
        if _type == "Edge" :
            for j in edges_clusters :
                if i in edges_clusters[j] : 
                    _egc = j 
                    break
            _egc = i
            print(_ram)
            print(_cpu)
            _pn = PN(i, _cpu,  _ram, _domain_id, type = _type, edge_cluster = _egc)
        else : 
            _pn = PN(i, _cpu,  _ram, _domain_id, type = _type)
        _PNS[i] = _pn

        _PLS = {}
    for j in range(len(nsparse_adj_list)):
        src , dist = nsparse_adj_list[j]
        _type = None 
        if (nodes_type[src] == 0 and nodes_type[dist] == 1) or (nodes_type[src] == 1 and nodes_type[dist] == 0):
            _type = "Edge"
        elif (nodes_type[src] == 0 and nodes_type[dist] == 0) or (nodes_type[src] == 0 and nodes_type[dist] == 2) or (nodes_type[src] == 2 and nodes_type[dist] == 0): 
            _type = "Transport"
        elif (nodes_type[src] == 2 and nodes_type[dist] == 2) : 
            _type = "Core"


        _p1 = _PNS[src]
        _p2 = _PNS[dist]

        _bw = 100#random.choice(DEFAULT_PL[_type]["BW"])
        _delay = random.choice(DEFAULT_PL[_type]["Delay"])
        _pl = PL(_p1, _p2 ,_bw,_delay)
        _PLS[j] = _pl

    _domain = Domain(0, "D", adj_mat, nsparse_adj_list, _PNS, _PLS)

    _domain.build_adj_attirbutes().shape
    adj = _domain.build_adj_attirbutes()
    attrs = _domain.get_attributes_pns()
    _domaines = {
        _domain._id :  _domain
    }

    MLFLOW_TRACKING_URI = "http://127.0.0.1:5005/"
    #PM = PlacementModule(_domaines, RL_boundaries, DEFAULT_VNF, DEFAULT_VL, edges_clusters)
    mlflow_volume = "/home/ryad/rl-multi-domain-for-multi-placement/vnf_placement/PlacementModule/volume_mlflow/mlflow"
    model_name = "salim" 

    tuner = FineTuner(
        boundaries= RL_boundaries,
        placemet_module= None,
        domaines = _domaines,
        DEFAULT_VNF = DEFAULT_VNF,
        DEFAULT_VL = DEFAULT_VL,
        edges_clusters = edges_clusters, 
        mlflow_url= MLFLOW_TRACKING_URI,
        model_name= model_name,
        mlflow_volume = mlflow_volume
    )

    tuner.tune(400)