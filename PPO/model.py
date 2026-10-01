from typing import Optional

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers, GaussianHead, CriticHead, CategoricalHead
from RLAlg.nn.steps import ValueStep, StochasticContinuousPolicyStep, DiscretePolicyStep

class ContinuousActor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], max_action:Optional[float | torch.Tensor]=None, norm_position:NormPosition=NormPosition.POST,
                 *, std_parameterization:str="log_std", init_std:float=1.0):
        super().__init__()

        # Use the requested normalization placement.
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        # Bound actions with a scaled tanh transform when max_action is supplied.
        self.head = GaussianHead(
            feature_dim, action_dim, max_action=max_action, state_dependent_std=False,
            std_parameterization=std_parameterization, init_std=init_std,
        )

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]=None) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x, action)

        return step
    
class DiscreteActor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        # Use the requested normalization placement.
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        self.head = CategoricalHead(feature_dim, action_dim)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]=None) -> DiscretePolicyStep:
        x = self.layers(x)

        step:DiscretePolicyStep = self.head(x, action)

        return step
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        # Use the requested normalization placement.
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        self.head = CriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> ValueStep:
        x = self.layers(x)

        step:ValueStep = self.head(x)

        return step
