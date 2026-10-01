from typing import Optional

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers, GaussianHead, DistributeCriticHead
from RLAlg.nn.steps import DistributionStep, StochasticContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[float | torch.Tensor]=None, norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        # Use the requested normalization placement.
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        # Bound actions with a scaled tanh transform when max_action is supplied.
        self.head = GaussianHead(feature_dim, action_dim, max_action=max_action, state_dependent_std=True)

    def forward(self, x:torch.Tensor) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x)

        return step
    
class QNet(nn.Module):
    def __init__(self, in_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        # Use the requested normalization placement.
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        self.head = DistributeCriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> DistributionStep:
        x = self.layers(x)

        step:DistributionStep = self.head(x)

        return step
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()
        
        self.critic_1 = QNet(in_dim+action_dim, hidden_dims, norm_position=norm_position)
        self.critic_2 = QNet(in_dim+action_dim, hidden_dims, norm_position=norm_position)

    def forward(self, x:torch.Tensor, action:torch.Tensor) -> tuple[DistributionStep, DistributionStep]:
        x = torch.cat([x, action], dim=-1)

        step_1:DistributionStep = self.critic_1(x)
        step_2:DistributionStep = self.critic_2(x)

        return step_1, step_2
