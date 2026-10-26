import math

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers, GaussianHead, CriticHead
from RLAlg.nn.steps import ValueStep, StochasticContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], init_std:float=math.exp(-1.0), norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.ELU(), norm_position=norm_position)

        # Reparameterized tanh-squashed samples keep the action differentiable for SHAC.
        self.head = GaussianHead(feature_dim, action_dim, init_std=init_std, max_action=1.0)

    def forward(self, x:torch.Tensor) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x)

        return step

class Critic(nn.Module):
    def __init__(self, in_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.ELU(), norm_position=norm_position)

        self.head = CriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> ValueStep:
        x = self.layers(x)

        step:ValueStep = self.head(x)

        return step
