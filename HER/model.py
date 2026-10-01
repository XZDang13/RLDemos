from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, DeterministicHead, CriticHead, NormPosition
from RLAlg.nn.steps import ValueStep, DeterministicContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[int]=None, norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = DeterministicHead(feature_dim, action_dim, max_action=max_action)

    def forward(self, x:dict[str:torch.Tensor]) -> DeterministicContinuousPolicyStep:
        obs = x["observation"]
        goal = x["desired_goal"]
        x = torch.cat([obs, goal], dim=1)
        
        x = self.layers(x)

        step:DeterministicContinuousPolicyStep = self.head(x)

        return step
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()
    
        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim+action_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        self.head = CriticHead(feature_dim)

    def forward(self, x:dict[str:torch.Tensor], action:torch.Tensor) -> ValueStep:
        obs = x["observation"]
        goal = x["desired_goal"]
        x = torch.cat([obs, goal, action], dim=1)
        
        x = self.layers(x)

        step:ValueStep = self.head(x)

        return step