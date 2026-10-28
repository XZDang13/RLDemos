import math

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers, GaussianHead, CriticHead
from RLAlg.nn.steps import ValueStep, StochasticContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        # State-dependent log-std in [-5, 2] and tanh squashing, as in the reference SAPO actor.
        # step.entropy is the one-sample estimate -log_prob of the reparameterized action.
        self.head = GaussianHead(feature_dim, action_dim, max_action=1.0, state_dependent_std=True,
                                 std_min=math.exp(-5.0), std_max=math.exp(2.0))

    def forward(self, x:torch.Tensor) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x)

        return step

class EnsembleCritic(nn.Module):
    """Independent value networks; SAPO bootstraps the policy from their mean and the targets from their minimum."""

    def __init__(self, in_dim:int, hidden_dims:list[int], num_critics:int=2, norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(num_critics):
            layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)
            self.layers.append(layers)
            self.heads.append(CriticHead(feature_dim))

    def forward(self, x:torch.Tensor) -> tuple[ValueStep, ...]:
        return tuple(head(layers(x)) for layers, head in zip(self.layers, self.heads))
