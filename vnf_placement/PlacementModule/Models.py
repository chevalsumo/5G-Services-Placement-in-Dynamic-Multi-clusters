import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
from torch_geometric.nn import ChebConv, ARMAConv


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == "Infra" :
                infra_size = subspace.shape[0]
                extractors[key] = Infra_Sequential(
                    ChebConv(subspace.shape[1],64,5, node_dim=1),
                    #ChebConv(subspace.shape[1],64,5),
                    nn.Flatten(start_dim=1,end_dim=-1),
                    #nn.ReLU()
                )
                total_concat_size += 64 * infra_size 
            elif key == "Vnf" :
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 128),
                    nn.Flatten(start_dim=1,end_dim=-1),
                )
                total_concat_size += 128
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim =  512
        self.infra_combine = nn.Sequential(
        nn.Linear(infra_size *  64 + 128,  self._features_dim),
        
        #nn.Linear(infra_size *  64 , self._features_dim),
        )
        
        print("Total :"+str(self._features_dim))
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "Infra":
                """
                ChebConv take as input the nodes attributes, Adjacency indices 
                and adjacency attributes
                """
                args = [observations[key],observations["Adj-List"],None] 
                infra_fts = extractor(*args).squeeze(0)
                encoded_tensor_list.append(infra_fts)
            elif key == "Vnf" :
                res = extractor(observations[key])
                encoded_tensor_list.append(res)

        result = th.cat(encoded_tensor_list, dim=1)
        
        result = result.squeeze(0)
        result = self.infra_combine(result)
        return result
    

class Infra_Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                b = []
                for x in inputs:
                    b.append(x)

                b[1] = b[1][0,:,:].squeeze(0)
                b[1] = b[1].type(th.int64)
                inputs = module(b[0],b[1],None)
            else:
                inputs = module(inputs)

        return inputs.unsqueeze(0)


class SimpleCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == "Infra" :
                infra_size = subspace.shape[0]

                extractors[key] = nn.Sequential(
                    nn.Flatten(start_dim=1,end_dim=-1),
                    nn.Linear(subspace.shape[0] * subspace.shape[1], 128),
                    nn.Flatten(start_dim=1,end_dim=-1),
                )
                """
                extractors[key] = Infra_Sequential(
                    ChebConv(subspace.shape[1],64,5, node_dim=1),
                    #ChebConv(subspace.shape[1],64,5),
                    nn.Flatten(start_dim=1,end_dim=-1),
                    #nn.ReLU()
                )                
                """

                total_concat_size += 64 * infra_size 
            elif key == "Vnf" :
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 32),
                    nn.Flatten(start_dim=1,end_dim=-1),
                )
                total_concat_size += 128
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim =  256
        self.infra_combine = nn.Sequential(
        nn.Linear(32 + 128,  self._features_dim),
        
        #nn.Linear(infra_size *  64 , self._features_dim),
        )
        
        print("Total :"+str(self._features_dim))

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "Infra":
                infra_fts = extractor(observations[key])
                encoded_tensor_list.append(infra_fts)
            elif key == "Vnf" :
                res = extractor(observations[key])
                encoded_tensor_list.append(res)

        result = th.cat(encoded_tensor_list, dim=1)
        
        result = result.squeeze(0)
        result = self.infra_combine(result)
        return result      