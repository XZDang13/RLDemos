import math

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers, GaussianHead, CriticHead
from RLAlg.nn.steps import ValueStep, StochasticContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], init_std:float=math.exp(-1.0), norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.ELU(), norm_position=norm_position)

        # Reparameterized tanh-squashed samples keep the action differentiable for AHAC.
        self.head = GaussianHead(feature_dim, action_dim, init_std=init_std, max_action=1.0)

    def forward(self, x:torch.Tensor) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x)

        return step

class DoubleCritic(nn.Module):
    """Two independent value networks; AHAC bootstraps from their minimum."""

    def __init__(self, in_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers1, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.ELU(), norm_position=norm_position)
        self.layers2, _ = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.ELU(), norm_position=norm_position)

        self.head1 = CriticHead(feature_dim)
        self.head2 = CriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> tuple[ValueStep, ValueStep]:
        step1:ValueStep = self.head1(self.layers1(x))
        step2:ValueStep = self.head2(self.layers2(x))

        return step1, step2
