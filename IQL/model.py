import math

import torch.nn as nn

from RLAlg.nn.layers import GaussianHead, NormPosition, make_mlp_layers

from PPO.model import Critic as Value
from SAC.model import Critic


class Actor(nn.Module):
    """Evaluate the probability of dataset actions for weighted imitation."""

    def __init__(self, in_dim, action_dim, hidden_dims, max_action):
        super().__init__()
        self.layers, feature_dim = make_mlp_layers(
            in_dim, hidden_dims, activate_function=nn.SiLU(),
            norm_position=NormPosition.POST,
        )
        self.head = GaussianHead(
            feature_dim, action_dim, max_action=max_action,
            state_dependent_std=True, std_min=math.exp(-5), std_max=math.exp(2),
        )

    def forward(self, obs, action=None):
        return self.head(self.layers(obs), action)
