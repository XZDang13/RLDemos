import math

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers, GaussianHead
from RLAlg.nn.steps import StochasticContinuousPolicyStep

# RPO uses SAPO's two live critics; train.py puts the repository root on sys.path.
from SAPO.model import EnsembleCritic

__all__ = ["Actor", "EnsembleCritic"]

class Actor(nn.Module):
    def __init__(self, in_dim:int, action_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        # The SAPO actor: tanh-squashed Gaussian with state-dependent log-std in [-5, 2].
        self.head = GaussianHead(feature_dim, action_dim, max_action=1.0, state_dependent_std=True,
                                 std_min=math.exp(-5.0), std_max=math.exp(2.0))

    def forward(self, x:torch.Tensor, action:torch.Tensor|None=None) -> StochasticContinuousPolicyStep:
        # With an action, the step scores that stored action (RPO's policy epochs).
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x, action)

        return step
