from typing import Optional

import torch
import torch.nn as nn

from RLAlg.nn.layers import (
    CategoricalDistributeCriticHead,
    GaussianHead,
    NormPosition,
    make_mlp_layers,
)
from RLAlg.nn.steps import CategoricalDistributionStep, StochasticContinuousPolicyStep


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        max_action: Optional[int] = None,
        norm_position: NormPosition = NormPosition.POST,
    ):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(
            in_dim,
            hidden_dims,
            activate_function=nn.SiLU(),
            norm_position=norm_position,
        )

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = GaussianHead(
            feature_dim,
            action_dim,
            max_action=max_action,
            state_dependent_std=True,
        )

    def forward(self, x: torch.Tensor) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step: StochasticContinuousPolicyStep = self.head(x)

        return step


class QNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        num_atoms: int = 51,
        v_min: float = -1000.0,
        v_max: float = 1000.0,
        norm_position: NormPosition = NormPosition.POST,
    ):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(
            in_dim,
            hidden_dims,
            activate_function=nn.SiLU(),
            norm_position=norm_position,
        )

        self.head = CategoricalDistributeCriticHead(
            feature_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
        )

    def forward(self, x: torch.Tensor) -> CategoricalDistributionStep:
        x = self.layers(x)

        step: CategoricalDistributionStep = self.head(x)

        return step


class Critic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        num_atoms: int = 51,
        v_min: float = -1000.0,
        v_max: float = 1000.0,
        norm_position: NormPosition = NormPosition.POST,
    ):
        super().__init__()

        self.critic_1 = QNet(
            in_dim + action_dim,
            hidden_dims,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            norm_position=norm_position,
        )
        self.critic_2 = QNet(
            in_dim + action_dim,
            hidden_dims,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            norm_position=norm_position,
        )

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[CategoricalDistributionStep, CategoricalDistributionStep]:
        x = torch.cat([x, action], dim=1)

        step_1: CategoricalDistributionStep = self.critic_1(x)
        step_2: CategoricalDistributionStep = self.critic_2(x)

        return step_1, step_2
