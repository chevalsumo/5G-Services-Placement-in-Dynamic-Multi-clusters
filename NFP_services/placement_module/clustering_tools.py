import numpy as np
import copy
from queue import PriorityQueue
from sklearn.cluster import SpectralClustering
import random

def matrix_to_sparse_adjacency(matrix):
    sparse_adjacency = []
    
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            if matrix[i][j] != 0:
                sparse_adjacency.append([i,j])
    
    return sparse_adjacency

def define_topology():
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

    num_nodes = len(adj_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            adj_matrix[node, neighbor] = 1
            adj_matrix[neighbor, node ] = 1

    nsparse_adj_list = matrix_to_sparse_adjacency(adj_matrix)

    nodes_type = [1, 0, 0, 2, 1, 0, 2, 2, 0, 1, 0]
    edges_clusters = {  0 : [0, 4, 9] }
    _adj_attrs = np.zeros((num_nodes, num_nodes, 2))
    _adj_attrs[:, :, 0] = 100
    for j in range(len(nsparse_adj_list)):
        src , dist = nsparse_adj_list[j]
        _type = None 
        if (nodes_type[src] == 0 and nodes_type[dist] == 1) or (nodes_type[src] == 1 and nodes_type[dist] == 0):
            _type = "Edge"
        elif (nodes_type[src] == 0 and nodes_type[dist] == 0) or (nodes_type[src] == 0 and nodes_type[dist] == 2) or (nodes_type[src] == 2 and nodes_type[dist] == 0): 
            _type = "Transport"
        elif (nodes_type[src] == 2 and nodes_type[dist] == 2) : 
            _type = "Core"

        _adj_attrs[src, dist, 1] = random.choice(DEFAULT_PL[_type]["Delay"])
        _adj_attrs[dist, src, 1] = random.choice(DEFAULT_PL[_type]["Delay"])
    return adj_matrix, nsparse_adj_list, _adj_attrs, edges_clusters

class Domain: 
    def __init__(self, _id, _type, _adj_matrix, _adj_list ,_PNS, _PLS, _adj_attrs = None, _pns_attrs = None, _deployed = False):
        # Domain 
        self._id = _id 
        self._type = _type 

        self._adj_matrix = _adj_matrix
        self._adj_list = _adj_list
        if (_PNS is not None) and (_PLS is not None):
        # In the case where we are in simulation and we get attributes from the created objects
            self._PLS = _PLS
            self._PNS = _PNS 
            self._adj_attrs = self.build_adj_attirbutes()
            self._pns_attrs = self.build_attributes_pns()
                        
        else : 
        # In the case of emulation where wet get attributes from collected metrics
            self._adj_attrs = _adj_attrs
            self.nb_nodes = self._adj_matrix.shape[0]


        if not _deployed : 
            # Create a copy of original attirbutes for the reset phase in the case that we are in the training phase
            self._pns_attrs = _pns_attrs
            self._org_pns_attributes = copy.deepcopy(self._pns_attrs)
            self._org_adj_list = copy.deepcopy(self._adj_attrs)

            self._max_cpu = self._org_pns_attributes[:,0].sum()
            self._max_ram = self._org_pns_attributes[:,1].sum()
            self._max_bw  = self._adj_attrs[:, :, 0].sum()



    def get_domain_load(self, _max_cpu, _max_ram, _max_bw):
        _load_cpu = _max_cpu / self._max_cpu
        _load_ram = _max_ram / self._max_ram
        _load_bw  = _max_bw  / self._max_bw
        return 1.0 - _load_cpu, 1.0 - _load_ram, 1.0 - _load_bw
    
    def get_adj_matrix(self):
        return self._adj_matrix
    
    def get_adj_list(self):
        return self._adj_list

    def get_node_neighbors(self, node):
        return list(np.where(self._adj_matrix[node, :])[0])
    
    def get_link_attrs(self, _src, _dist):
        #Get attributes of a link with source and distination nodes 
        return self._adj_attrs[_src, _dist, :]

    def build_adj_attirbutes(self):
        #Number of ressources that we are gonna consider 
        nb_ressources = 2  #BW/Delay 
        nb_nodes = len(self._PNS)
        self.nb_nodes = nb_nodes

        adj_attrs = np.zeros((nb_nodes, nb_nodes, nb_ressources))
        for _, _pl in self._PLS.items() :
            atrs = _pl.get_attributes()
            src, dist = [int(i) for i in _pl.get_id().split("/")]
            adj_attrs[src, dist] = atrs
            adj_attrs[dist, src] = atrs
        return adj_attrs

 
    def get_PNS(self):
        return self._PNS

    def get_PLS(self):
        return self._PLS

    def get_nb_nodes(self):
        return self.nb_nodes
    def get_nodes_degree_BW(self):
        #Get the degree and bandwidth available in the links connected to the nodes 
        degrees = self._adj_matrix.sum(axis=0)
        nodes_BW = self._adj_attrs.sum(axis= 0)[:,0]
        return degrees, nodes_BW
    
    def build_attributes_pns(self, inc_load = True, inc_degree = False , inc_BW = True, inc_delay = False):
        #Build matrix of physical nodes attributes 
        result = None
        for i, _pn in self._PNS.items():
            _pn_atrs = _pn.get_attributes(load = inc_load)
            if result is None : 
                result= _pn_atrs
            else : 
                result = np.vstack((result,_pn_atrs))

        degrees, nodes_BW = self.get_nodes_degree_BW()
        if inc_BW : 
            result = np.column_stack((result, nodes_BW))
        if inc_degree : 
            result = np.column_stack((result, nodes_BW, degrees))
        return result.astype(np.float32)
    
    def set_attributes_pns(self, _pns_attrs):
        self._pns_attrs = _pns_attrs 

    def set_attributes_pls(self, _pls_attrs):
        self._adj_attrs = _pls_attrs

    def get_attributes_pns(self, actual = True):
        #Get matrix of physical nodes attributes 
        if actual : 
            return self._pns_attrs
        else : 
            #else return orignal attributes for a reset
            return self._org_pns_attributes

    def dijkstra(self, _source, mask_delay = False, mask_bw = False, required_BW = None, required_delay = None, MAX_DEFAULT_NODE_DELAY = float("inf")): 
        #Dijkstra algorithom to find the shortest path from _source to all the other nodes
        #D = {v : float("inf") for v in range(self.nb_nodes)}
        D = np.full((self.nb_nodes,), float(MAX_DEFAULT_NODE_DELAY))
        paths = {v : [] for v in range(self.nb_nodes)}
        pq = PriorityQueue()
        pq.put((0, _source))
        visited = []
        D[_source] = 0
        while not pq.empty() : 
            (dist, current_vertex) = pq.get()
            visited.append(current_vertex)
            neighbors = self.get_node_neighbors(current_vertex)
            for j in list(neighbors) : 
                if j not in visited :
                    pass_mask = True 
                    if mask_bw :
                            if required_BW > self._adj_attrs[current_vertex, j , 0]  :
                                pass_mask = False    
                    if  pass_mask : 
                        old_cost = D[j]
                        new_cost = D[current_vertex] +  self._adj_attrs[current_vertex, j , 1]
                        if (new_cost < old_cost) :
                            pass_mask = True 
                            if (mask_delay) and (new_cost > required_delay) :
                                pass_mask = False 
                            if pass_mask :
                                pq.put((new_cost,j))
                                D[j] = new_cost
                                paths[j] = paths[current_vertex].copy()
                                paths[j].append(j)
   
        if mask_bw or mask_delay :
            mask = (D != MAX_DEFAULT_NODE_DELAY)
            return D, paths, mask
        else : 
            return D, paths, None
        

class AugmentedGraph:

    def __init__(self, nb_cluster):
        self._nb_cluster = nb_cluster 
        self._clusters = None

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