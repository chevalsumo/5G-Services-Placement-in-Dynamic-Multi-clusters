import numpy as np
import random 
import requests 

class VNF: 
    def __init__(self, _id, _rqr_CPU, _rqr_MEMORY):
        self._id = _id
        self._rqr_CPU = _rqr_CPU
        self._rqr_MEMORY = _rqr_MEMORY

    def get_attributes(self):

        return np.array([self._rqr_CPU, self._rqr_MEMORY])
    
    def __repr__(self):
        s = "VNF: "+str(self._id)+"(" + str(self.get_attributes())+")"
        return s
class VL: 
    def __init__(self, _id, VNF1, VNF2, _rqr_BW, _rqr_DELAY):
        self._id = _id 
        #VNF source
        self._vnf_s = VNF1
        #VNF destination 
        self._vnf_d = VNF2
        #Required bandwith 
        self._rqr_BW = _rqr_BW
        #Required delay 
        self._rqr_DELAY = _rqr_DELAY 
    
    def get_vnf_source(self):
        return self._vnf_s

    def get_vnf_destination(self):
        return self._vnf_d 
    
    def get_attributes(self, BW = True, DELAY = True):
        if BW and DELAY : 
            return np.array([self._rqr_BW, DELAY])
        elif BW : 
            return np.array([self._rqr_BW])
        elif DELAY : 
            return np.array([self._rqr_DELAY])
    def __repr__(self):
        s = "ID:" + self._id + " Attrs: " + str(self.get_attributes(BW= False))
        return s
class E2E_Service_Chain:
    #La chaine VNFs répresentant le service 
    def __init__(self, _id, VNF_init, _VNFs, _VLs, _VLs_Attrs, _sources):
        self._id = _id 
        self._vnf_init = VNF_init
        self._vnfs = _VNFs
        self._VLs = _VLs
        self._VLs_Attrs = _VLs_Attrs
        self._sources = _sources
    def get_VNFs_BW_delay(self):
        return self._VLs_Attrs[:,:,1:].sum(axis = 0)[1:]
    def get_VNFs_attrs(self, vl_attrs = True, BW = False):
    #Get the attributes of VNFs
        result = None
        if BW:
            _node_vls_attrs = self.get_VNFs_BW_delay()
        else:
            _node_vls_attrs = self.get_VNFs_BW_delay()[:,1].reshape(-1,1)
        for _, _vnf in self._vnfs.items() : 
            _vnf_attrs = _vnf.get_attributes()
            if result is None : 
                result= _vnf_attrs
            else : 
                result = np.vstack((result, _vnf_attrs))
        if vl_attrs :
            result = np.hstack((result, _node_vls_attrs))
        return result
    def get_init_vnf(self): 
        return self._vnf_init

    
    def __repr__(self):
        #s = "ID:" + self._id + " Attrs: " + str(self.get_attributes())
        return repr(list(self._vnfs.keys()))#self
    

class Slice_request : 
    #Slice request 
    def __init__(self, _id = None, init = "rand", edge = "rand" ,SERVICE_chain = None , DEFAULT_VNF = None , DEFAULT_VL = None, edge_clusters = None, edge_node = "rand", min_nb_vnfs = 3, max_nb_vnfs = 10):
        
        self._id = _id 
        if init == None : 
            self._chain_service = SERVICE_chain
            """
            Normally, the entry point is the cluster or GnB to which the user 
            is connected, but since in our case we're simulating the RAN part 
            and even the UE and GnB represent NFs to be placed, 
            we'll take it at random. 
            """
            if edge == "rand" : 
                _edge_cluster = random.choice(list(edge_clusters.keys()))
                self._enter_nodes = edge_clusters[_edge_cluster]
                #Chose which edge node to use for user 
                if edge_node == "rand" :
                    self._enter_node = random.choice(self._enter_nodes)
        elif init == "rand" :
            #Generate random service chain of 3-10 VNFs 
            nb_vnfs = random.randint(min_nb_vnfs, max_nb_vnfs)
            _vnfs = {}

            _vls_mat = np.zeros((nb_vnfs + 1, nb_vnfs + 1, 3))
            _vls = {}
            #Chose which edge area (domain for later) to use 
            if edge == "rand" : 
                _edge_cluster = random.choice(list(edge_clusters.keys()))
                self._enter_nodes = edge_clusters[_edge_cluster]
                #Chose which edge node to use for user 
                if edge_node == "rand" :
                    self._enter_node = random.choice(self._enter_nodes)
                elif edge_node == "best" : 
                    pass
            for j in range(1, nb_vnfs + 1): 
                #VNFs of the service chain 
                _cpu = random.choice(DEFAULT_VNF["CPU"])
                _ram = random.choice(DEFAULT_VNF["RAM"])
                _vnf = VNF(j, _cpu, _ram)
                _vnfs[j] = _vnf
                #VLs of the service chain 
                _bw = random.choice(DEFAULT_VL["BW"])
                _delay = DEFAULT_VL["Delay"][0] if j <= 2 else DEFAULT_VL["Delay"][1]
                _vl  = VL(j, j-1, j, _bw, _delay)
                #Adjacency matrix
                _vls_mat[j-1,j][0] = 1
                _vls_mat[j-1,j][1:] = _bw, _delay
                #If we wanna constraint one VNF per node _vls_mat[j,j- 1][1] = _bw
                """
                Delay and bandwidth matrix Directional VNFR?
                _vls_mat[j,j -1][0] = 1
                _vls_mat[j,j- 1][1:] = _bw, _delay Directional VNFR?
                """


                _vls[j-1] = _vl 
            _id = 0 
            _sc = E2E_Service_Chain(_id, _vnfs[1],_vnfs, _vls,_vls_mat)
            self._chain_service = _sc
        elif init == "5g":
            #_types = ["CN-RAN", "CN", "RAN"]
            _types = ["v2x"]#["CN-RAN"]
            _type = random.choice(_types)
            rg_api_call = requests.get(f"http://localhost:8000/rg/doc/{self._id}/{_type}") # API CALL
            _res = rg_api_call.json()
            sr = _res["doc"]
            _delays = _res["delay"]

            #_ue = sr["nfs"]["ue"] 
            #sr["nfs"].pop("ue")

            _nfs = list(sr["nfs"].keys())

            _order = ["ue", "nrf" ,"amf", "gnb", "upf", "DN"]
            _sorted_keys  = _order +  [key for key in _nfs if key not in _order]
            
            _VNFs = {}

            _Vls = {}
            _vls_mat = np.zeros((len(sr["nfs"]), len(sr["nfs"]), 3))
            _sources = {}
            for idx, nf in enumerate(_sorted_keys):
                _cpu = sr["nfs"][nf]['cpu_limits'] / 1000
                _ram = sr["nfs"][nf]['ram_limits']
                if nf != "ue":

                    _VNFs[nf] = VNF(nf, _cpu, _ram)
                    _delay = _delays[nf]["delay"]
                    _source = _sorted_keys.index(_delays[nf]["source"])
                    _bw = 0
                    _vls_mat[_source, idx][0] = 1
                    _vls_mat[_source, idx][1:] = _bw, _delay 
                    _Vls[idx] = VL(f"{_sorted_keys[_source]}/{_sorted_keys[idx]}", _sorted_keys[_source], _sorted_keys[idx], _bw, _delay)
                    _sources[idx] = _source
                else:
                    _init_vnf = VNF(nf, _cpu, _ram)  
            self._chain_service = E2E_Service_Chain(_id, _init_vnf, _VNFs, _Vls, _vls_mat, _sources = _sources)  
            """
                _Vls = {}
                _vls_mat = np.zeros((len(sr["nfs"]) + 1, len(sr["nfs"]) + 1, 3))
                for i in range(len(sr["nfs"])):
                    for j in range(i + 1, len(sr["nfs"])):
                        _bw = 0 
                        _delay = 100
                        _vls_mat[i,j][0] = 1
                        _vls_mat[i,j][1:] = _bw, _delay
                        _Vls[i+j] = VL(f"{_nfs[i]}/{_nfs[j]}", _nfs[i], _nfs[j], _bw, _delay) 
                self._chain_service = E2E_Service_Chain(_id, _nfs[0], _VNFs, _Vls, _vls_mat)
            """
            if edge == "rand" : 
                _edge_cluster = random.choice(list(edge_clusters.keys()))
                self._enter_nodes = edge_clusters[_edge_cluster]
                #Chose which edge node to use for user 
                if edge_node == "rand" :
                    self._enter_node = random.choice(self._enter_nodes)
    def get_service_chain(self):
        #Get VNFs chain used for this request
        return self._chain_service

    def get_enter_node(self):
        #Get enter edge node user for this request 
        return self._enter_node
    
    def get_VNFR_attrs(self, vl_attrs = True):
        return self._chain_service.get_VNFs_attrs(vl_attrs)
    
    def __repr__(self):
        return repr(self._chain_service)#self