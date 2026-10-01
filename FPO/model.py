import torch
import torch.nn as nn

from RLAlg.nn.layers import DiffusionHead, NormPosition, make_mlp_layers


class NormalizedModel(nn.Module):
    """Use frozen observation statistics for both sampling and loss evaluation."""

    def __init__(self, model, normalizer):
        super().__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, obs, *args, **kwargs):
        mean = self.normalizer.mean.to(dtype=obs.dtype)
        std = self.normalizer.var.to(dtype=obs.dtype).sqrt().clamp_min(0.1)
        observations = ((obs - mean) / std).clamp(-10, 10)
        return self.model(observations, *args, **kwargs)


class Actor(nn.Module):
    """Predict a flow velocity from observations, latent actions, and time."""

    def __init__(self, in_dim, action_dim, hidden_dims):
        super().__init__()
        self.layers, feature_dim = make_mlp_layers(
            in_dim + action_dim + 1, hidden_dims,
            activate_function=nn.SiLU(), norm_position=NormPosition.POST,
        )
        self.head = DiffusionHead(feature_dim, action_dim)

    def forward(self, obs, action, time):
        # Works for both [batch, features] and [batch, CMF samples, features].
        features = torch.cat([obs, action, time], dim=-1)
        return self.head(self.layers(features))
