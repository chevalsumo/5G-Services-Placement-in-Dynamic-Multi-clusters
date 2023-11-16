import gym 
from gym import spaces
import numpy as np
import copy
import time


#Placement Gym envrionnement 
class PlacementEnv(gym.Env):
    def __init__(self, 
            _id_domain = None, boundaries = None, PModule = None, requests = None, 
            maskable = None, episode_type = "Request", mask_bw = True, mask_delay = False, 
            mask_cpu = False, mask_ram = False,reward_type = "Requirements",
             _reset_load_pourcent = 1, _test_mode = False,
             rew_st_s = 0, rew_st_f = 0, rew_chain_s = 0, rew_chain_f = 0, rew_rs_step_tries = None,
             verbose = 1, 
             ):
        super().__init__()

        self._verbose = verbose
        # Rewards 
        self._rew_st_s = rew_st_s
        self._rew_st_f = rew_st_f
        self._rew_chain_s = rew_chain_s
        self._rew_chain_f = rew_chain_f
        self._rew_rs_step_tries = rew_rs_step_tries
        self.boundaries = boundaries
        self.PModule = PModule
        self._id_domain = _id_domain
        #self._org_requests = copy.deepcopy(requests)
        self._requests = requests
        self._episode_type = episode_type
        self._mask_delay = mask_delay
        self._mask_bw = mask_bw
        self._mask_cpu = mask_cpu 
        self._mask_ram = mask_ram
        self._test_mode = _test_mode

        self._nb_actions = self.PModule.get_domain_nb_actions(self._id_domain)
        self._infra_adj_list = np.array(self.PModule.get_infra_adj_list(_id = _id_domain), dtype= np.int64).T
        self._reset_load_pourcent = _reset_load_pourcent

        values_keys = {
            "infra" : ["DEFAULT_CPU", "DEFAULT_RAM", "DEFAULT_NODE_DELAY"],
            "VNF"   : ["DEFAULT_VNFS_CPU","DEFAULT_VNFS_RAM", "SERVICES_DELAY"],
                      }
        #Define Observation space limits (Low, High values)
        hight_values_infra = np.array([self.boundaries["MAX_"+j] for j in values_keys["infra"]], dtype = np.float32)
        low_values_infra = np.array([self.boundaries["MIN_"+j] for j in values_keys["infra"]], dtype = np.float32)


        hight_values_infra = np.tile(hight_values_infra, (self._nb_actions,1))
        low_values_infra = np.tile(low_values_infra, (self._nb_actions,1))

        hight_values_vnf = np.array([self.boundaries["MAX_"+j] for j in values_keys["VNF"]], dtype = np.float32)
        low_values_vnf = np.array([self.boundaries["MIN_"+j] for j in values_keys["VNF"]], dtype = np.float32)

        #Define Observation space as Dict of two Boxes (Infra and VNFs)
        self.observation_space = gym.spaces.Dict(
            {
            "Infra" : gym.spaces.Box(low = low_values_infra, high = hight_values_infra),
            "Vnf"   : gym.spaces.Box(low = low_values_vnf, high = hight_values_vnf),           
            }
        )
        #Define action space with the number of actions as the number of physical nodes
        self.action_space = spaces.Discrete(self._nb_actions)
        self._id_request2place = 0
        self._reward_type = reward_type

        self._type_reset = "init"
        self._accp = 0
        self._ref  = 0
        self._ep_tries = 0
        self._step_tries = 0
    def reset(self):
        obs = self.reset_placement()
        return obs
    
    def reset_placement(self): 
        if self._requests is not None :
            if self._id_request2place < (len(self._requests)-1): 
                self._id_request2place += 1 
            #print(self._type_reset)
            else : 
                self._id_request2place = 0
                if self._episode_type == "Batch_Requests" :
                    self._type_reset = "load"
            _sr = self._requests[self._id_request2place]
            #print(f"Placement {_sr }")
        else : 
            _sr = self.PModule.create_random_request(_type = "5g")
        obs = self.build_reset_obs(_sr, reset_type= self._type_reset)
        #Save the infrastructre attributes (if the request is rejected)
        self._clean_pns_atrs, self._clean_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
        
        #k = copy.deepcopy(obs)
        #k["Infra"][:,0:3] -= k["Vnf"][0:3]
        #del k["Vnf"]
        return obs
            
    def step(self, action):
        accepted, reward, done = self.take_action(action, reward_type= self._reward_type)
        #print(f"Accepted: {accepted} Done: {done} Reward: {reward}") verbose
        if not done : 
        #If not done, we place the next VNF: 
            self.build_step_obs(action)
            if self._episode_type == "Request-successful" :
                self._step_tries = 0
                #print(list(self._requests[self._id_request2place].get_service_chain()._vnfs.keys())[self._vnf_to_place - 1])
                #self._ep_tries += 1
                #self._clean_step_pns_atrs, self._clean_step_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
                
        else : 
            if accepted : 
                #Reset with the next request without chaning infra attributes  
                self._type_reset = "keep_obs"
                self._accp += 1  
                
                #self._ep_tries = 0
                self._step_tries = 0
            
            else :
                if self._episode_type != "Request-successful" and self._episode_type != "Fail-VNF" :
                #Reset with the next request and restore observation space 
                    self._type_reset = "clean_obs"
                    self._ref  += 1 
                    if self._requests is not None :
                        if not self._test_mode : 
                            self._id_request2place = len(self._requests)-1
                else : 
                    self._step_tries += 1
                    self._ep_tries += 1
                    """
                    if self._sFail-VNFep_trietep_tries < self._rew_rs_step_tries :
                        done = False
                    else :
                        self._step_tries = 0   
                        print("Echeco")
                        self._possible_actions = np.logical_and(self.obs["Infra"][:, 1] >= self.obs["Vnf"][1], self.obs["Infra"][:, 0] >= self.obs["Vnf"][0])
                        if (self._possible_actions.sum() < 1) :
                            #print("Pas d'action possible connard")
                            self._type_reset = "load"
                    """
                    self._possible_actions = np.logical_and(self.obs["Infra"][:, 1] >= self.obs["Vnf"][1], self.obs["Infra"][:, 0] >= self.obs["Vnf"][0], self.obs["Infra"][:, 2] <= self.obs["Vnf"][2])
                    if (self._possible_actions.sum() < 1) :
                        print("Pas d'action possible connard")
                        self._type_reset = "load"  
                    elif (self._step_tries > 50):
                        #print("because hadi")
                        self._type_reset = "load"
                    else : 
                        #print(f"NB act:{self._possible_actions.sum()} act:{action} Reward: {reward}")
                        done = False              
                    #self._pns_atrs, self._pls_attrs = copy.deepcopy(self._clean_step_pns_atrs), copy.deepcopy(self._clean_step_pls_attrs)
            if self._episode_type == "Request" : 
                done = True
            elif ((self._episode_type == "Batch_Requests") or (self._episode_type == "Fail-VNF")) and (self._requests is None):
                if accepted :
                    done = False 
                    self.reset_placement()  
                else : 
                    #print(f"Dergana kbira 3lik {done}")
                    self._type_reset = "load"     
                    done = True              
            elif self._episode_type == "Batch_Requests":
                if (self._id_request2place == (len(self._requests)-1)):
                    done = True
                else : 
                    done = False 
                    self.reset_placement()
        #k = copy.deepcopy(self.obs)
        #k["Infra"][:,0:3] -= k["Vnf"][0:3]
        #del k["Vnf"]
        #print(reward)

        return self.obs, reward, done, {}

    
    def take_action(self, action, reward_type = "Requirements"):
        done = False 
        """
        If reward type is "Best", we reward the agent according to at which
        point the action is better than the others independently of the VNFs requirements 

        If reward type is "Requirements" the rewards are proportional to the degree of respect of the constraints
        """
        _path = self._paths[action]
        _cost = self._costs[action]
        reward = 0
        accepted = True 


        """
        Warning: We may have - inf for reward when we divise by the available ressources
        """
        #Checking Bandwidth and Delay constraints 
        if ((self._mask_delay or self._mask_bw) and _cost == self.boundaries["MAX_DEFAULT_NODE_DELAY"]) :
            #pass
            
            #pass
            
            # Use only CPU, RAM V1
            accepted = False 
            

            #print("Very Bad choice Delay/BW")

        #If we don't mask delay we can reward agent proportionally to this constraint 
        #elif not self._mask_delay :
        if reward_type == "Requirements" : 
            """
            if _cost > self.obs["Vnf"][3] :
                accepted = False
                _r_del = - _cost/self.obs["Vnf"][3]
            else :
                if np.isclose(_cost, 0.0) :
                    #Cost Zero, means we place the VNF in the same node as the last one, so we don't have latency
                    _r_del = self.boundaries['MAX_SERVICES_DELAY']  / self.boundaries['MIN_DEFAULT_DELAY'] + 0.5
                else :  
                    _r_del = + self.obs["Vnf"][3]/_cost
            """
        elif reward_type == "Best" :
            pass
            """
             # Use only CPU, RAM V1
            _r_del = 1 - _cost / max(self._costs)
            """
            
        # Checking CPU and RAM constraints
        # CPU 
        if reward_type == "Requirements" : 
            if self.obs["Infra"][action, 0] < self.obs["Vnf"][0]:
                accepted = False
                _r_cpu = -  self.obs["Vnf"][0] / self.obs["Infra"][action, 0]
            else : 
                _r_cpu = +  self.obs["Infra"][action, 0] / self.obs["Vnf"][0]
        elif reward_type == "Best" :
            _r_cpu = self.obs["Infra"][action, 0] / max(self.obs["Infra"][:, 0])
        # RAM
        if reward_type == "Requirements" : 
            if self.obs["Infra"][action, 1] < self.obs["Vnf"][1]:
                accepted = False
                _r_ram = -  self.obs["Vnf"][1] / self.obs["Infra"][action, 1]  
            else :
                _r_ram = +  self.obs["Infra"][action, 1] / self.obs["Vnf"][1]
        elif reward_type == "Best" :
            _r_ram = self.obs["Infra"][action, 1] / max(self.obs["Infra"][:, 1])
        #if not self._mask_delay :

        if reward_type == "Requirements" :
            print("petit test")
            #reward = 0.0 * (0.33 * _r_del + 0.33 * _r_cpu + 0.33 * _r_ram)
        elif reward_type == "Best" : 
            reward = self._rew_st_s #* (0.2 * _r_del + 0.4 * _r_cpu + 0.4 * _r_ram)
         
    
        # Use only CPU, RAM V1
        if (self.obs["Infra"][action, 0] < self.obs["Vnf"][0]) or (self.obs["Infra"][action, 1] < self.obs["Vnf"][1]) or (_cost > self.obs["Vnf"][2]) :
                accepted = False       
        """ 
        if (self.obs["Infra"][action, 0] < self.obs["Vnf"][0]) or (self.obs["Infra"][action, 1] < self.obs["Vnf"][1]):
                accepted = False 
        """ 

        """
        else : 
            reward = 100 * (0.5 * _r_cpu + 0.5 * _r_ram)
        """
        #Scale factor compared to the size of the VNFs chain
        if not accepted :
            #If the placement is not accepted (it's final state, so we penalize the agent)
            _sf =   (0.5 * (self.boundaries["MAX_SERVICES_VNFS"]+ self.boundaries["MIN_SERVICES_VNFS"])) / self._vnfs_to_place
            reward = self._rew_chain_f #* _sf 
            
            done = True
        elif (self._vnfs_to_place == self._vnf_to_place) :
            #If the placement of the last VNF of the request is accepted (it's final state)
            _sf = self._vnfs_to_place /  (0.5 * (self.boundaries["MAX_SERVICES_VNFS"]+ self.boundaries["MIN_SERVICES_VNFS"])) 
            reward += self._rew_chain_s #_sf
            done = True
        #print(done)
        if done : 
            if self._test_mode :
                print(f"REQ:({self._id_request2place})VNF: ({self._vnf_to_place}/{self._vnfs_to_place}) Reward: {reward} R_del: {_r_del} R_CPU: {_r_cpu} R_RAM {_r_ram}")
 
        #Update infrasturcure attributes and observation space if the placement is accepted
        if accepted : 
            #Update CPU and RAM
            #In observation space 
            self.obs["Infra"][action, 0:2] -= self.obs["Vnf"][0:2] 

            """
            # Use only CPU, RAM V1
            #Update BW
            _src = self._last_PN
            _bw = self.obs["Vnf"][2]
            #print(_path)
            for _dist in _path  :
                #print(f"{_src} {_dist} BW:{_bw}")
                self._pls_attrs[_src, _dist][0] -= _bw
                self._pls_attrs[_dist, _src][0] -= _bw
                self.obs["Infra"][_src, 2] -= _bw
                self.obs["Infra"][_dist, 2] -= _bw
                _src = _dist

            self._pns_atrs[:,0:3] = self.obs["Infra"][:,0:3]
            """
            #self._pns_atrs[:,0:3] = self.obs["Infra"][:,0:3]
            self._pns_atrs[:,0:2] = self.obs["Infra"][:,0:2] # Use only CPU, RAM V1
        
        #print(reward)
        return accepted, reward, done 

    def build_reset_obs(self, req, reset_type = "init"):
        obs = {}
        self._req = req
        self._req_atrs = self._req.get_VNFR_attrs()
        self._enter_node = self._req.get_enter_node()
        self._vnfs_to_place = self._req_atrs.shape[0]
        #Set the VNF to place to the first one 
        self._vnf_to_place = 1
        
        #Set the last seclected PN as the edge enter node
        self._last_PN = self._enter_node 

        #Get UE NF 
        self._ue_attrs = self._req._chain_service.get_init_vnf().get_attributes()

        #Get First VNF attributes 
        self._vnf_atrs = self._req_atrs[self._vnf_to_place - 1].astype(np.float32)
        if reset_type  != "init" :
            _cpu_load, _ram_load, _bw_load = self.get_domain_load(self._id_domain)
            #print(f"CPU: {_cpu_load} RAM: {_ram_load} BW: {_bw_load}")
            if _cpu_load > self._reset_load_pourcent or _ram_load > self._reset_load_pourcent:
                reset_type = "load"
        if reset_type == "init" :
            self._pns_atrs, self._pls_attrs = self.PModule.get_infra_attributes(_id =self._id_domain, inc_load = False, inc_degree = False, inc_BW = True, shuffle = True)
            self._init_pns_atrs, self._init_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
        elif reset_type == "clean_obs":
            #In this case,
            self._pns_atrs, self._pls_attrs = copy.deepcopy(self._clean_pns_atrs), copy.deepcopy(self._clean_pls_attrs)
        elif reset_type == "keep_obs":
            #In this case, we keep the observation space for the next placement 
            pass
        elif reset_type == "load" :
            self._pns_atrs, self._pls_attrs = self.PModule.get_infra_attributes(_id =self._id_domain, inc_load = False, inc_degree = False, inc_BW = True, shuffle = True)
            self._init_pns_atrs, self._init_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
            #self._pns_atrs, self._pls_attrs = copy.deepcopy(self._init_pns_atrs), copy.deepcopy(self._init_pls_attrs)
            if self._verbose: 
                print(f"CPU: {_cpu_load} RAM: {_ram_load} BW: {_bw_load}")
                print(f"ACP:{self._accp }, REF:{self._ref}, AR:{self._accp/(self._accp+self._ref) if (self._accp+self._ref) else 0}")
            #print(self._ep_tries)
            self._ep_tries = 0
            self._accp = 0
            self._ref  = 0
            self._step_tries = 0
        
        # Place the UE NF on the selected edge node (cluster)
        self._pns_atrs[self._enter_node,0:2] -= self._ue_attrs
        
        #Get dijkstra from last selected PN to all the other PNs
        self._costs, self._paths, self._mask = self.PModule._domaines[self._id_domain].dijkstra(self._last_PN, mask_bw = self._mask_bw, mask_delay = self._mask_delay, required_BW = None, required_delay = self._vnf_atrs[2], MAX_DEFAULT_NODE_DELAY = self.boundaries["MAX_DEFAULT_NODE_DELAY"])
        
        #Set observation space 

        # Use only CPU, RAM and delay [:, 0:2]

        obs["Infra"] = np.hstack((self._pns_atrs[:, 0:2], self._costs.reshape(-1, 1)))

        obs["Vnf"] = self._vnf_atrs
        obs["Adj-List"] = self._infra_adj_list

        # Use only CPU, RAM  
        
        #obs["Infra"] = obs["Infra"][:,0:2]
        #obs["Vnf"] = obs["Vnf"][0:2]

        
        self.obs = obs

        #CPU mask:
        if self._mask_cpu : 
            self._cpu_mask = self.obs["Infra"][:, 0] >= self.obs["Vnf"][0]
            self._mask = np.logical_and(self._mask, self._cpu_mask)

        #RAM mask:
        if self._mask_ram : 
            self._ram_mask = self.obs["Infra"][:, 1] >= self.obs["Vnf"][1]
            self._mask = np.logical_and(self._mask, self._ram_mask)

        # Check if ther is no longer possible action 
        self._possible_actions = np.logical_and(self.obs["Infra"][:, 1] >= self.obs["Vnf"][1], self.obs["Infra"][:, 0] >= self.obs["Vnf"][0])
        if (self._mask.sum() < 1) or (self._possible_actions.sum() < 1) :
            #print("Pas d'action possible connard")
            pass

        return obs 
    
    def build_step_obs(self, action):
        #Prepare the observation spaec for the next VNF placement 

        self._vnf_to_place += 1 
        self._vnf_atrs = self._req_atrs[self._vnf_to_place - 1]
        #Set the last physical node to the action that we select (For the next placement)
        self._last_PN = action

        #Get dijkstra from last selected PN to all the other PNs
        self._costs, self._paths, self._mask = self.PModule._domaines[self._id_domain].dijkstra(self._last_PN, mask_bw = self._mask_bw, mask_delay = self._mask_delay, required_BW = None, required_delay = self._vnf_atrs[2], MAX_DEFAULT_NODE_DELAY = self.boundaries["MAX_DEFAULT_NODE_DELAY"])
        
        #print(self._costs)        
        #print(self._paths)

        #Set observation space 
    
        # Use only CPU, RAM V1  and Dijkstra Costs 
        self.obs["Infra"][:, 2] = self._costs
    
        self.obs["Vnf"] = self._vnf_atrs
        self.obs["Adj-List"] = self._infra_adj_list

        #CPU mask:
        if self._mask_cpu : 
            self._cpu_mask = self.obs["Infra"][:, 0] >= self.obs["Vnf"][0]
            self._mask = np.logical_and(self._mask, self._cpu_mask)

        #RAM mask:
        if self._mask_ram : 
            self._ram_mask = self.obs["Infra"][:, 1] >= self.obs["Vnf"][1]
            self._mask = np.logical_and(self._mask, self._ram_mask)

        if self._mask.sum() < 1 :
            #print("Pas d'action possible connard")
            pass

        # Use only CPU, RAM V1
        #self.obs["Infra"] = self.obs["Infra"][:,0:2] # Use only CPU, RAM V1
        #self.obs["Vnf"] = self.obs["Vnf"][0:2] # Use only CPU, RAM V1

    def get_domain_load(self, _domain_id):


        _cpu = self.obs["Infra"][:, 0].sum()
        _ram = self.obs["Infra"][:, 1].sum()
        _bw  = self._pls_attrs[:, :, 0].sum()

        _load_cpu, _load_ram, _load_bw = self.PModule._domaines[self._id_domain].get_domain_load(_cpu, _ram, _bw)
        return _load_cpu, _load_ram, _load_bw 

    def action_masks(self):
        return list(self._mask)


class ClusEnv(gym.Env):
    def __init__(self, 
            _id_domain = None, boundaries = None, PModule = None, requests = None, 
            maskable = None, episode_type = "Request", mask_bw = True, mask_delay = False, 
            mask_cpu = False, mask_ram = False,reward_type = "Requirements",
             _reset_load_pourcent = 1, _test_mode = False,
             rew_st_s = 0, rew_st_f = 0, rew_chain_s = 0, rew_chain_f = 0, rew_rs_step_tries = None,
             verbose = 1, 
             clustering = False,
             nb_cluster = False,
             keep_clusters = True
             ):
        super().__init__()

        self._verbose = verbose
        # Rewards 
        self._rew_st_s = rew_st_s
        self._rew_st_f = rew_st_f
        self._rew_chain_s = rew_chain_s
        self._rew_chain_f = rew_chain_f
        self._rew_rs_step_tries = rew_rs_step_tries
        self.boundaries = boundaries
        self.PModule = PModule
        self._id_domain = _id_domain
        #self._org_requests = copy.deepcopy(requests)
        self._requests = requests
        self._episode_type = episode_type
        self._mask_delay = mask_delay
        self._mask_bw = mask_bw
        self._mask_cpu = mask_cpu 
        self._mask_ram = mask_ram
        self._test_mode = _test_mode
        self._clustering = clustering 
        self._nb_cluster = nb_cluster
        self._keep_clusters = keep_clusters
        self._nb_actions = self.PModule.get_domain_nb_actions(self._id_domain)
        self._infra_adj_list = np.array(self.PModule.get_infra_adj_list(_id = _id_domain), dtype= np.int64).T
        self._reset_load_pourcent = _reset_load_pourcent

        values_keys = {
            "infra" : ["DEFAULT_CPU", "DEFAULT_RAM", "DEFAULT_NODE_DELAY"],
            "VNF"   : ["DEFAULT_VNFS_CPU","DEFAULT_VNFS_RAM", "SERVICES_DELAY"],
                      }
        #Define Observation space limits (Low, High values)
        hight_values_infra = np.array([self.boundaries["MAX_"+j] for j in values_keys["infra"]], dtype = np.float32)
        low_values_infra = np.array([self.boundaries["MIN_"+j] for j in values_keys["infra"]], dtype = np.float32)


        hight_values_infra = np.tile(hight_values_infra, (self._nb_actions,1))
        low_values_infra = np.tile(low_values_infra, (self._nb_actions,1))

        hight_values_vnf = np.array([self.boundaries["MAX_"+j] for j in values_keys["VNF"]], dtype = np.float32)
        low_values_vnf = np.array([self.boundaries["MIN_"+j] for j in values_keys["VNF"]], dtype = np.float32)

        #Define Observation space as Dict of two Boxes (Infra and VNFs)
        self.observation_space = gym.spaces.Dict(
            {
            "Infra" : gym.spaces.Box(low = low_values_infra, high = hight_values_infra),
            "Vnf"   : gym.spaces.Box(low = low_values_vnf, high = hight_values_vnf),           
            }
        )
        #Define action space with the number of actions as the number of physical nodes
        self.action_space = spaces.Discrete(self._nb_actions)
        self._id_request2place = 0
        self._reward_type = reward_type

        self._type_reset = "init"
        self._accp = 0
        self._ref  = 0
        self._ep_tries = 0
        self._step_tries = 0
    def reset(self):
        obs = self.reset_placement()
        return obs
    
    def reset_placement(self): 
        if self._requests is not None :
            if self._id_request2place < (len(self._requests)-1): 
                self._id_request2place += 1 
            #print(self._type_reset)
            else : 
                self._id_request2place = 0
                if self._episode_type == "Batch_Requests" :
                    self._type_reset = "load"
            _sr = self._requests[self._id_request2place]
            #print(f"Placement {_sr }")
        else : 
            _sr = self.PModule.create_random_request(_type = "5g")
        obs = self.build_reset_obs(_sr, reset_type= self._type_reset)
        #Save the infrastructre attributes (if the request is rejected)
        
        #self._clean_pns_atrs, self._clean_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)


        return obs
            
    def step(self, action):
        #print("Before Step")
        #print(self._costs)
        accepted, reward, done = self.take_action(action, reward_type= self._reward_type)
        #print("After Step")
        #print(self._pns_atrs)
        #print(f"Accepted: {accepted} Done: {done} Reward: {reward}") verbose
        if not done : 
        #If not done, we place the next VNF: 
            self.build_step_obs(action)
            if self._episode_type == "Request-successful" :
                self._step_tries = 0
                #print(list(self._requests[self._id_request2place].get_service_chain()._vnfs.keys())[self._vnf_to_place - 1])
                #self._ep_tries += 1
                #self._clean_step_pns_atrs, self._clean_step_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
                
        else : 
            if accepted : 
                #Reset with the next request without chaning infra attributes  
                self._type_reset = "keep_obs"
                self._accp += 1  
                
                #self._ep_tries = 0
                self._step_tries = 0
            
            else :
                if self._episode_type != "Request-successful" and self._episode_type != "Fail-VNF" :
                #Reset with the next request and restore observation space 
                    self._type_reset = "clean_obs"
                    self._ref  += 1 
                    if self._requests is not None :
                        if not self._test_mode : 
                            self._id_request2place = len(self._requests)-1
                else : 
                    self._step_tries += 1
                    self._ep_tries += 1
                    """
                    if self._sFail-VNFep_trietep_tries < self._rew_rs_step_tries :
                        done = False
                    else :
                        self._step_tries = 0   
                        print("Echeco")
                        self._possible_actions = np.logical_and(self.obs["Infra"][:, 1] >= self.obs["Vnf"][1], self.obs["Infra"][:, 0] >= self.obs["Vnf"][0])
                        if (self._possible_actions.sum() < 1) :
                            #print("Pas d'action possible connard")
                            self._type_reset = "load"
                    """
                    self._possible_actions = np.logical_and(self.obs["Infra"][:, 1] >= self.obs["Vnf"][1], self.obs["Infra"][:, 0] >= self.obs["Vnf"][0], self.obs["Infra"][:, 2] <= self.obs["Vnf"][2])
                    if (self._possible_actions.sum() < 1) :
                        print("Pas d'action possible connard")
                        self._type_reset = "load"  
                    elif (self._step_tries > 50):
                        print("because hadi")
                        self._type_reset = "load"
                    else : 
                        #print(f"NB act:{self._possible_actions.sum()} act:{action} Reward: {reward}")
                        done = False              
                    #self._pns_atrs, self._pls_attrs = copy.deepcopy(self._clean_step_pns_atrs), copy.deepcopy(self._clean_step_pls_attrs)
            if self._episode_type == "Request" : 
                done = True
            elif ((self._episode_type == "Batch_Requests") or (self._episode_type == "Fail-VNF")) and (self._requests is None):
                if accepted :
                    done = False 
                    self.reset_placement()  
                else : 
                    #print(f"Dergana kbira 3lik {done}")
                    self._type_reset = "load"     
                    done = True              
            elif self._episode_type == "Batch_Requests":
                if (self._id_request2place == (len(self._requests)-1)):
                    done = True
                else : 
                    done = False 
                    self.reset_placement()
        #k = copy.deepcopy(self.obs)
        #k["Infra"][:,0:3] -= k["Vnf"][0:3]
        #del k["Vnf"]
        #print(reward)

        return self.obs, reward, done, {}

    
    def take_action(self, action, reward_type = "Requirements"):
        done = False 
        """
        If reward type is "Best", we reward the agent according to at which
        point the action is better than the others independently of the VNFs requirements 

        If reward type is "Requirements" the rewards are proportional to the degree of respect of the constraints
        """
        if self._clustering :
            final_action = self.PModule.get_final_action(action, _choice = "Max_RAM", _vnf_obs = self.obs["Vnf"])
            #print(f"Final Action : {final_action}")
            _path = self._paths[final_action]
            _cost = self._costs[final_action]
            self._final_action = final_action

        else:
            _path = self._paths[action]
            _cost = self._costs[action]
        reward = 0
        accepted = True 


        """
        Warning: We may have - inf for reward when we divise by the available ressources
        """
        #Checking Bandwidth and Delay constraints 
        if ((self._mask_delay or self._mask_bw) and _cost == self.boundaries["MAX_DEFAULT_NODE_DELAY"]) :
            #pass
            
            #pass
            
            # Use only CPU, RAM V1
            accepted = False 
            

            #print("Very Bad choice Delay/BW")

        #If we don't mask delay we can reward agent proportionally to this constraint 
        #elif not self._mask_delay :
        if reward_type == "Requirements" : 
            """
            if _cost > self.obs["Vnf"][3] :
                accepted = False
                _r_del = - _cost/self.obs["Vnf"][3]
            else :
                if np.isclose(_cost, 0.0) :
                    #Cost Zero, means we place the VNF in the same node as the last one, so we don't have latency
                    _r_del = self.boundaries['MAX_SERVICES_DELAY']  / self.boundaries['MIN_DEFAULT_DELAY'] + 0.5
                else :  
                    _r_del = + self.obs["Vnf"][3]/_cost
            """
        elif reward_type == "Best" :
            pass
            """
             # Use only CPU, RAM V1
            _r_del = 1 - _cost / max(self._costs)
            """
            
        # Checking CPU and RAM constraints
        # CPU 
        if reward_type == "Requirements" : 
            if self.obs["Infra"][action, 0] < self.obs["Vnf"][0]:
                accepted = False
                _r_cpu = -  self.obs["Vnf"][0] / self.obs["Infra"][action, 0]
            else : 
                _r_cpu = +  self.obs["Infra"][action, 0] / self.obs["Vnf"][0]
        elif reward_type == "Best" :
            _r_cpu = self.obs["Infra"][action, 0] / max(self.obs["Infra"][:, 0])
        # RAM
        if reward_type == "Requirements" : 
            if self.obs["Infra"][action, 1] < self.obs["Vnf"][1]:
                accepted = False
                _r_ram = -  self.obs["Vnf"][1] / self.obs["Infra"][action, 1]  
            else :
                _r_ram = +  self.obs["Infra"][action, 1] / self.obs["Vnf"][1]
        elif reward_type == "Best" :
            _r_ram = self.obs["Infra"][action, 1] / max(self.obs["Infra"][:, 1])
        #if not self._mask_delay :

        if reward_type == "Requirements" :
            print("petit test")
            #reward = 0.0 * (0.33 * _r_del + 0.33 * _r_cpu + 0.33 * _r_ram)
        elif reward_type == "Best" : 
            reward = self._rew_st_s #* (0.2 * _r_del + 0.4 * _r_cpu + 0.4 * _r_ram)
         
    
        # Use only CPU, RAM V1
        if (self.obs["Infra"][action, 0] < self.obs["Vnf"][0]) or (self.obs["Infra"][action, 1] < self.obs["Vnf"][1]) or (_cost > self.obs["Vnf"][2]) :
                accepted = False       
        """ 
        if (self.obs["Infra"][action, 0] < self.obs["Vnf"][0]) or (self.obs["Infra"][action, 1] < self.obs["Vnf"][1]):
                accepted = False 
        """ 

        """
        else : 
            reward = 100 * (0.5 * _r_cpu + 0.5 * _r_ram)
        """
        #Scale factor compared to the size of the VNFs chain
        if not accepted:
            #If the placement is not accepted (it's final state, so we penalize the agent)
            _sf =   (0.5 * (self.boundaries["MAX_SERVICES_VNFS"]+ self.boundaries["MIN_SERVICES_VNFS"])) / self._vnfs_to_place
            reward = self._rew_chain_f #* _sf 
            
            done = True
        elif (self._vnfs_to_place == self._vnf_to_place) :
            #If the placement of the last VNF of the request is accepted (it's final state)
            _sf = self._vnfs_to_place /  (0.5 * (self.boundaries["MAX_SERVICES_VNFS"]+ self.boundaries["MIN_SERVICES_VNFS"])) 
            reward += self._rew_chain_s #_sf
            done = True
        #print(done)
        if done : 
            if self._test_mode :
                print(f"REQ:({self._id_request2place})VNF: ({self._vnf_to_place}/{self._vnfs_to_place}) Reward: {reward} R_del: {_r_del} R_CPU: {_r_cpu} R_RAM {_r_ram}")
 
        #Update infrasturcure attributes and observation space if the placement is accepted
        if accepted : 
            #Update CPU and RAM
            #In observation space 
            if not self._clustering :
                self.obs["Infra"][action, 0:2] -= self.obs["Vnf"][0:2] 
                #self._pns_atrs[:,0:3] = self.obs["Infra"][:,0:3]
                self._pns_atrs[:,0:2] = self.obs["Infra"][:,0:2] # Use only CPU, RAM V1
            else :
                self._pns_atrs[final_action,0:2] -= self.obs["Vnf"][0:2]
            """
            # Use only CPU, RAM V1
            #Update BW
            _src = self._last_PN
            _bw = self.obs["Vnf"][2]
            #print(_path)
            for _dist in _path  :
                #print(f"{_src} {_dist} BW:{_bw}")
                self._pls_attrs[_src, _dist][0] -= _bw
                self._pls_attrs[_dist, _src][0] -= _bw
                self.obs["Infra"][_src, 2] -= _bw
                self.obs["Infra"][_dist, 2] -= _bw
                _src = _dist

            self._pns_atrs[:,0:3] = self.obs["Infra"][:,0:3]
            """

        
        #print(reward)
        return accepted, reward, done 

    def build_reset_obs(self, req, reset_type = "init"):
        obs = {}
        self._req = req
        self._req_atrs = self._req.get_VNFR_attrs()
        
        self._enter_node = self._req.get_enter_node()
        self._vnfs_to_place = self._req_atrs.shape[0]

        self._sources = self._req._chain_service._sources
        self._placement = {}
        #Set the VNF to place to the first one 
        self._vnf_to_place = 1
        
        #Set the last seclected PN as the edge enter node
        self._last_PN = self._enter_node 
        self._placement[0] = self._enter_node 
        #Get UE NF 
        self._ue_attrs = self._req._chain_service.get_init_vnf().get_attributes()

        #Get First VNF attributes 
        self._vnf_atrs = self._req_atrs[self._vnf_to_place - 1].astype(np.float32)
        #print(f"Reset Type {reset_type}")
        if reset_type  != "init" :
            _cpu_load, _ram_load, _bw_load = self.get_domain_load(self._id_domain)
            #print(f"CPU: {_cpu_load} RAM: {_ram_load} BW: {_bw_load}")
            #print(f"RAM load {_ram_load} CPU load {_cpu_load}")
            if _cpu_load > self._reset_load_pourcent or _ram_load > self._reset_load_pourcent:
                reset_type = "load"
        if reset_type == "init" :
            self._pns_atrs, self._pls_attrs = self.PModule.get_infra_attributes(_id =self._id_domain, inc_load = False, inc_degree = False, inc_BW = True, shuffle = True)
            self._init_pns_atrs, self._init_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
        elif reset_type == "clean_obs":
            #In this case,
            self._pns_atrs, self._pls_attrs = copy.deepcopy(self._clean_pns_atrs), copy.deepcopy(self._clean_pls_attrs)
        elif reset_type == "keep_obs":
            #In this case, we keep the observation space for the next placement 
            pass
        elif reset_type == "load" :
            self._pns_atrs, self._pls_attrs = self.PModule.get_infra_attributes(_id =self._id_domain, inc_load = False, inc_degree = False, inc_BW = True, shuffle = True)
            self._init_pns_atrs, self._init_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)
            #self._pns_atrs, self._pls_attrs = copy.deepcopy(self._init_pns_atrs), copy.deepcopy(self._init_pls_attrs)
            if self._verbose: 
                print(f"CPU: {_cpu_load} RAM: {_ram_load} BW: {_bw_load}")
                #print(f"ACP:{self._accp }, REF:{self._ref}, AR:{self._accp/(self._accp+self._ref) if (self._accp+self._ref) else 0}")
            #print(self._ep_tries)
            self._ep_tries = 0
            self._accp = 0
            self._ref  = 0
            self._step_tries = 0
        
        self._clean_pns_atrs, self._clean_pls_attrs = copy.deepcopy(self._pns_atrs), copy.deepcopy(self._pls_attrs)

        if self._clustering:
            self._cpu_load, self._ram_load, _bw_load = self.get_domain_load(self._id_domain)
            #print(f"RAM load {self._ram_load} CPU load {self._cpu_load}")
        
            # Place the UE NF on the selected edge node (cluster)
            self._pns_atrs[self._enter_node,0:2] -= self._ue_attrs
        
        #Get dijkstra from last selected PN to all the other PNs
        self._costs, self._paths, self._mask = self.PModule._domaines[self._id_domain].dijkstra(self._last_PN, mask_bw = self._mask_bw, mask_delay = self._mask_delay, required_BW = None, required_delay = self._vnf_atrs[2], MAX_DEFAULT_NODE_DELAY = self.boundaries["MAX_DEFAULT_NODE_DELAY"])
        
        #Set observation space 

        # Use only CPU, RAM and delay [:, 0:2]
        if not self._clustering:
            obs["Infra"] = np.hstack((self._pns_atrs[:, 0:2], self._costs.reshape(-1, 1)))
            
            obs["Adj-List"] = self._infra_adj_list
        else :  
            obs["Infra"] = self.PModule.get_clusters_attrs(self._pls_attrs, self._pns_atrs, self._costs, choice = "Min_DELAY", keep_clusters = self._keep_clusters)

        obs["Vnf"] = self._vnf_atrs
        # Use only CPU, RAM  
        
        #obs["Infra"] = obs["Infra"][:,0:2]
        #obs["Vnf"] = obs["Vnf"][0:2]

        
        self.obs = obs
        if not self._clustering:
            self._cpu_load, self._ram_load, _bw_load = self.get_domain_load(self._id_domain)

        #CPU mask:
        if self._mask_cpu : 
            self._cpu_mask = self.obs["Infra"][:, 0] >= self.obs["Vnf"][0]
            self._mask = np.logical_and(self._mask, self._cpu_mask)

        #RAM mask:
        if self._mask_ram : 
            self._ram_mask = self.obs["Infra"][:, 1] >= self.obs["Vnf"][1]
            self._mask = np.logical_and(self._mask, self._ram_mask)

        # Check if ther is no longer possible action 
        self._possible_actions = np.logical_and(self.obs["Infra"][:, 1] >= self.obs["Vnf"][1], self.obs["Infra"][:, 0] >= self.obs["Vnf"][0])
        if (self._mask.sum() < 1) or (self._possible_actions.sum() < 1) :
            #print("Pas d'action possible connard")
            pass
        
        return obs 
    
    def build_step_obs(self, action):
        #Prepare the observation spaec for the next VNF placement 

        self._vnf_to_place += 1 
        self._vnf_atrs = self._req_atrs[self._vnf_to_place - 1]
        #Set the last physical node to the action that we select (For the next placement)
        if self._clustering:
            self._placement[self._vnf_to_place - 1] = self._final_action
            #self._last_PN = self._sources[]
        else:
            self._placement[self._vnf_to_place - 1] = action
        #print(f"Placement dans: {self._final_action}")
        self._last_PN = self._placement[self._sources[self._vnf_to_place]]
        
        #print(f"VNF to place {self._vnf_to_place}, la source {self._last_PN} placement de la VNF {self._sources[self._vnf_to_place]}")
        #Get dijkstra from last selected PN to all the other PNs
        self._costs, self._paths, self._mask = self.PModule._domaines[self._id_domain].dijkstra(self._last_PN, mask_bw = self._mask_bw, mask_delay = self._mask_delay, required_BW = None, required_delay = self._vnf_atrs[2], MAX_DEFAULT_NODE_DELAY = self.boundaries["MAX_DEFAULT_NODE_DELAY"])
        

        #Set observation space 
    
        # Use only CPU, RAM V1  and Dijkstra Costs
         
        
        if not self._clustering:
            self.obs["Infra"][:, 2] = self._costs
            self.obs["Adj-List"] = self._infra_adj_list
        else :  
            self.obs["Infra"] = self.PModule.get_clusters_attrs(self._pls_attrs, self._pns_atrs, self._costs, choice = "Min_DELAY", keep_clusters = self._keep_clusters)

        #print(self._pns_atrs)
        self.obs["Vnf"] = self._vnf_atrs
        

        #CPU mask:
        if self._mask_cpu : 
            self._cpu_mask = self.obs["Infra"][:, 0] >= self.obs["Vnf"][0]
            self._mask = np.logical_and(self._mask, self._cpu_mask)

        #RAM mask:
        if self._mask_ram : 
            self._ram_mask = self.obs["Infra"][:, 1] >= self.obs["Vnf"][1]
            self._mask = np.logical_and(self._mask, self._ram_mask)

        if self._mask.sum() < 1 :
            #print("Pas d'action possible connard")
            pass

        # Use only CPU, RAM V1
        #self.obs["Infra"] = self.obs["Infra"][:,0:2] # Use only CPU, RAM V1
        #self.obs["Vnf"] = self.obs["Vnf"][0:2] # Use only CPU, RAM V1


    def get_domain_load(self, _domain_id):

        if self._clustering:
            _cpu = self._pns_atrs[:, 0].sum()
            _ram = self._pns_atrs[:, 1].sum()

        else :
            _cpu = self.obs["Infra"][:, 0].sum()
            _ram = self.obs["Infra"][:, 1].sum()
        _bw  = self._pls_attrs[:, :, 0].sum()

        _load_cpu, _load_ram, _load_bw = self.PModule._domaines[self._id_domain].get_domain_load(_cpu, _ram, _bw)
        return _load_cpu, _load_ram, _load_bw 

    def action_masks(self):
        return list(self._mask)


