from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, GaussianHead, DistributeCriticHead
from RLAlg.nn.steps import DistributionStep, StochasticContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[int]=None):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm=True)

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = GaussianHead(feature_dim, action_dim, max_action=max_action, state_dependent_std=True)

    def forward(self, x:torch.Tensor) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x)

        return step
    
class QNet(nn.Module):
    def __init__(self, in_dim:int, hidden_dims:list[int]):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm=True)

        self.head = DistributeCriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> DistributionStep:
        x = self.layers(x)

        step:DistributionStep = self.head(x)

        return step
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int]):
        super().__init__()
        
        self.critic_1 = QNet(in_dim+action_dim, hidden_dims)
        self.critic_2 = QNet(in_dim+action_dim, hidden_dims)

    def forward(self, x:torch.Tensor, action:torch.Tensor) -> tuple[DistributionStep, DistributionStep]:
        x = torch.cat([x, action], dim=1)

        step_1:DistributionStep = self.critic_1(x)
        step_2:DistributionStep = self.critic_2(x)

        return step_1, step_2