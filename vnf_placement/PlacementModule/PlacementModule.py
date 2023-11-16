from vnf_placement.VNFR.VNFR import * 
from sklearn.cluster import SpectralClustering

#Placement module class 
class PlacementModule:
    def __init__(self, _domaines, _boundaries, DEFAULT_VNF, DEFAULT_VL, _edges_clusters, _init_load, _clustering = False, _nb_cluster = None, _clusters = None):
        self._boundaries = _boundaries
        self._domaines = _domaines
        self._DEFAULT_VNF = DEFAULT_VNF
        self._DEFAULT_VL = DEFAULT_VL
        self._edges_clusters = _edges_clusters
        self._init_load = _init_load
        self._clustering = _clustering
        #self._init_attrs = _init_attrs
        if _clustering: 
            
            self._nb_clustering = _nb_cluster
            self._clusters = _clusters
            self._augGraph= AugmentedGraph(self._nb_clustering, _clusters)
    def get_domaines(self):
        return self._domaines
    

    def get_clusters_attrs(self, _adj_attrs, _pns_attrs, _costs, choice = "Min_DELAY", keep_clusters = True):
        _cls_attrs, _cls_costs = self._augGraph.get_augmented_graph(_adj_attrs = _adj_attrs[:,:,1], _pns_attrs = _pns_attrs, _costs= _costs, MAX_DEFAULT_NODE_DELAY=  self._boundaries["MAX_DEFAULT_NODE_DELAY"], choice = choice, keep_clusters= keep_clusters)
        _cls_obs = np.hstack((_cls_attrs[:, 0:2], _cls_costs.reshape(-1, 1)))
        return _cls_obs
    
    def get_final_action(self, _selected_cluster, _choice = "Min_DELAY", _vnf_obs = None):
        return self._augGraph.get_final_action(_selected_cluster, MAX_DEFAULT_NODE_DELAY= self._boundaries["MAX_DEFAULT_NODE_DELAY"], choice = _choice)
    
    def get_domaine(self,id):
        return self._domaines[id]
    
    def get_infra_attributes(self, _id = 0, inc_load = False, inc_degree = False, inc_BW = True, shuffle = False, clustering = False, production = False):
        _domain = self._domaines[_id]
        if not production :
            _pns_attrs = _domain.build_attributes_pns(inc_load = inc_load, inc_degree = inc_degree, inc_BW = inc_BW)
            _pls_attrs = _domain.build_adj_attirbutes()
        else : 
            _pns_attrs = _domain._pns_attrs
            _pls_attrs = _domain._adj_attrs
        if shuffle:
            _pns_attrs = self.generate_random_resource_configuration(self._init_load, _pns_attrs, 5)
        else :
            _domain.set_attributes_pns(_pns_attrs)

        _domain.set_attributes_pls(_pls_attrs)
        if clustering:
            pass
        return _pns_attrs, _pls_attrs
    
    def get_infra_adj_list(self, _id = 0):
        return  self._domaines[_id].get_adj_list()
    
    def get_domain_nb_actions(self, _id = 0):
        #Get number of actions in a domain (number of nodes) 
        if self._clustering:
            return self._nb_clustering
        else:   
            return self._domaines[_id].get_nb_nodes()
    
    def create_random_request(self, _type = "rand"):
        if _type ==  "rand":
            return Slice_request(init= "rand", edge = "rand", DEFAULT_VNF= self._DEFAULT_VNF, DEFAULT_VL = self._DEFAULT_VL, edge_clusters = self._edges_clusters, edge_node = "rand", min_nb_vnfs = self._boundaries["MIN_SERVICES_VNFS"], max_nb_vnfs = self._boundaries["MAX_SERVICES_VNFS"])
        elif _type == "5g":
            return Slice_request(init= "5g", edge= "rand", edge_clusters = self._edges_clusters)

    def generate_random_resource_configuration(self, percentage, clusters, min_per):
        num_clusters = clusters.shape[0]
        remain = np.zeros((num_clusters, 3)) 
        
        remain[:, 1] = clusters[:, 1] * (1 - percentage/100) * min_per/100 
        
        total_ram = np.sum(clusters[:, 1]) * (1 - percentage/100) - np.sum(remain[:, 1])
        available_clusters = list(range(num_clusters))
        
        while total_ram > 1:
            cls = random.choice(available_clusters)
            available_ram_cls = remain[cls, 1]
            total_ram_cls = total_ram / num_clusters
            add_ram = random.uniform(total_ram_cls / 3, total_ram_cls * 3)
            
            if (available_ram_cls + add_ram) < clusters[cls, 1]:
                remain[cls, 1] += add_ram
                total_ram -= add_ram
            
            available_clusters.remove(cls)
            if not available_clusters:
                available_clusters = list(range(num_clusters))

        remain[:, 0] = clusters[:, 0] * (1 - (1 - remain[:, 1] / clusters[:, 1]) * 0.53)
        remain[:, 2] = clusters[:, 2]
        
        return remain
    
    


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