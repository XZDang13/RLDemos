from typing import Optional

import torch
import torch.nn as nn

from RLAlg.nn.layers import make_mlp_layers, GaussianHead, CriticHead, CategoricalHead, NormPosition, GRULayer
from RLAlg.nn.steps import ValueStep, StochasticContinuousPolicyStep, DiscretePolicyStep

class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        gru_num_layers: int = 1,
        norm_position: NormPosition = NormPosition.NONE,
    ):
        super().__init__()
        if gru_num_layers <= 0:
            raise ValueError(f"gru_num_layers must be positive, got {gru_num_layers}.")
        
        self.layers, self.feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)
        self.gru_num_layers = gru_num_layers
        self.recurrent_layer = GRULayer(self.feature_dim, self.feature_dim, num_layers=gru_num_layers)

    @property
    def hidden_state_shape(self) -> tuple[int, ...]:
        if self.gru_num_layers == 1:
            return (self.feature_dim,)
        return (self.gru_num_layers, self.feature_dim)

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        hidden = torch.zeros(batch_size, self.gru_num_layers, self.feature_dim, device=device)
        if self.gru_num_layers == 1:
            return hidden.squeeze(1)
        return hidden

    def _to_gru_hidden(self, h: torch.Tensor, batch_size: int, x: torch.Tensor) -> torch.Tensor:
        expected_bh = (batch_size, self.feature_dim)
        expected_blh = (batch_size, self.gru_num_layers, self.feature_dim)
        expected_lbh = (self.gru_num_layers, batch_size, self.feature_dim)
        if h.ndim == 2:
            if self.gru_num_layers != 1 or h.shape != expected_bh:
                raise ValueError(
                    f"For gru_num_layers={self.gru_num_layers}, hidden_state must be {expected_blh} or {expected_lbh}, got {tuple(h.shape)}."
                )
            gru_hidden = h.unsqueeze(0)
        elif h.ndim == 3:
            if h.shape == expected_blh:
                gru_hidden = h.permute(1, 0, 2).contiguous()
            elif h.shape == expected_lbh:
                gru_hidden = h
            else:
                raise ValueError(
                    f"hidden_state must be {expected_blh} or {expected_lbh}, got {tuple(h.shape)}."
                )
        else:
            raise ValueError(
                f"hidden_state must be 2D/3D tensor, got shape {tuple(h.shape)}."
            )
        return gru_hidden.to(dtype=x.dtype, device=x.device)
        
        
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)
        gru_hidden = None
        if h is not None:
            batch_size = x.shape[0] if x.ndim == 2 else x.shape[1]
            gru_hidden = self._to_gru_hidden(h, batch_size, x)
        x, next_hidden = self.recurrent_layer(x, hidden_state=gru_hidden, episode_starts=episode_starts)
        next_hidden = next_hidden.permute(1, 0, 2).contiguous()
        if self.gru_num_layers == 1:
            next_hidden = next_hidden.squeeze(1)
        return x, next_hidden

class ContinuousActor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        max_action: Optional[torch.Tensor] = None,
        norm_position: NormPosition = NormPosition.PRE,
    ):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(
            in_dim,
            hidden_dims,
            activate_function=nn.SiLU(),
            norm_position=norm_position,
        )

        #if max action is setted, normal distribution will be scaled tanh transformed.
        #if state_dependent_std is True, log std will be learned from obs
        self.head = GaussianHead(feature_dim, action_dim, max_action=max_action, state_dependent_std=False)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]=None) -> StochasticContinuousPolicyStep:
        x = self.layers(x)

        step:StochasticContinuousPolicyStep = self.head(x, action)

        return step
    
class DiscreteActor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        norm_position: NormPosition = NormPosition.PRE,
    ):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(
            in_dim,
            hidden_dims,
            activate_function=nn.SiLU(),
            norm_position=norm_position,
        )

        self.head = CategoricalHead(feature_dim, action_dim)

    def forward(self, x:torch.Tensor, action:Optional[torch.Tensor]=None) -> DiscretePolicyStep:
        x = self.layers(x)

        step:DiscretePolicyStep = self.head(x, action)

        return step
    
class Critic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        norm_position: NormPosition = NormPosition.PRE,
    ):
        super().__init__()

        self.layers, feature_dim = make_mlp_layers(
            in_dim,
            hidden_dims,
            activate_function=nn.SiLU(),
            norm_position=norm_position,
        )

        self.head = CriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> ValueStep:
        x = self.layers(x)

        step:ValueStep = self.head(x)

        return step
