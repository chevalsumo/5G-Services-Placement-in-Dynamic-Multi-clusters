import numpy as np
import copy
from queue import PriorityQueue

#Physical Link 
class PL: 
    def __init__(self, _PN1, _PN2, MAX_BW, INDUCED_DELAY):
        #Physical link attributes 
        self._id = str(_PN1.get_id()) + "/" + str(_PN2.get_id())
        self._id_back = str(_PN2.get_id()) + "/" + str(_PN1.get_id())

        self._pn_s = _PN1.get_id()
        self._pn_d = _PN2.get_id()   

        self._max_BW = MAX_BW
        self._remain_BW = MAX_BW
    
        self._induced_delay = INDUCED_DELAY
    def check_node_pl(self, _pn):
        #Check this PL is connected to _pn
        return (self._pn_s == _pn) or (self._pn_d == _pn)

    def get_bw_load(self):
        #Check BW load 
        return float("{:.3f}".format(1 - self._remain_BW / self._max_BW))
    
    def get_attributes(self):
        #Get attribute of the physical link
        return np.array([self._remain_BW, self._induced_delay])

    def get_id(self):
        return self._id
    
    def get_id_back(self):
        return self._id_back

    def __repr__(self):
        s = "ID:" + self._id + " Attrs: " + str(self.get_attributes())
        return s


#Physical Node 
class PN:
    def __init__(self,_id, MAX_CPU, MAX_RAM, domain_ID, x_coord = None, y_coord = None, type = None, edge_cluster = None ):
        #Physical node attributes 
        self._id = _id 
        self._x_coord = x_coord
        self._y_coord = y_coord

        self._domain_id = domain_ID

        self._max_cpu = MAX_CPU
        self._max_ram = MAX_RAM 

        self._remain_cpu = MAX_CPU
        self._remain_ram = MAX_RAM

        self._type = type
        self._edge_cluster = edge_cluster

    def get_ressource_load(self, ressource):
        #Load of ressources in the PN
        if ressource == "CPU" : 
            return float("{:.3f}".format(1 - self._remain_cpu / self._max_cpu))
        elif ressource == "RAM":
            return float("{:.3f}".format(1 - self._remain_ram / self._max_ram))

    def update_ressources(self, _CPU, _RAM, allocation = True):
        #Allocate or Release resources 
        if allocation : 
            self._remain_cpu -= _CPU
            self._remain_ram -= _RAM
        else : 
            self._remain_cpu += _CPU
            self._remain_ram += _RAM  

    def get_attributes(self, load = True):
        if not load : 
            return np.array([self._remain_cpu, self._remain_ram])
        else : 
            return np.array([self._remain_cpu, self._remain_ram, self.get_ressource_load("CPU"), self.get_ressource_load("RAM")])

    def get_id(self):
        return self._id

    
class Domain: 
    def __init__(self, _id, _type, _adj_matrix, _adj_list ,_PNS, _PLS, _adj_attrs = None, _pns_attrs = None):
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
            self._pns_attrs = _pns_attrs
            self.nb_nodes = _pns_attrs.shape[0]



        # Create a copy of original attirbutes for the reset phase
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

