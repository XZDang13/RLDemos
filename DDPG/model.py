from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, DeterministicHead, CriticHead
from RLAlg.nn.steps import ValueStep, DeterministicContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[int]=None):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm=True)

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = DeterministicHead(feature_dim, action_dim, max_action=max_action)

    def forward(self, x:torch.Tensor) -> DeterministicContinuousPolicyStep:
        x = self.layers(x)

        step:DeterministicContinuousPolicyStep = self.head(x)

        return step
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int]):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim+action_dim, hidden_dims, activate_function=nn.SiLU(), norm=True)

        self.head = CriticHead(feature_dim)

    def forward(self, x:torch.Tensor, action:torch.Tensor) -> ValueStep:
        x = torch.cat([x, action], dim=1)
        
        x = self.layers(x)

        step:ValueStep = self.head(x)

        return step