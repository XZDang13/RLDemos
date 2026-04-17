from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, MLPLayer, DiffusionHead, CriticHead, NormPosition
from RLAlg.nn.steps import ValueStep



class Actor(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int, embedding_dim:int, hidden_dims:list[int],
                 velocity_scale:float, norm_position: NormPosition.POST):
        super().__init__()
        
        #self.obs_embedding = MLPLayer(obs_dim, embedding_dim, nn.SiLU(), norm_position)
        #self.action_embedding = MLPLayer(action_dim, embedding_dim, nn.SiLU(), norm_position)
        #self.time_embedding = MLPLayer(1, embedding_dim, nn.SiLU(), norm_position)
        
        embedding_dim = obs_dim + action_dim + 8
        
        self.layer, feature_dim = make_mlp_layers(embedding_dim, hidden_dims, nn.SiLU(), norm_position)
        
        self.head = DiffusionHead(feature_dim, action_dim, velocity_scale)
    
    def embed_timestep(self, t: torch.Tensor, timestep_embed_dim: int) -> torch.Tensor:
            """
            Embed (*, 1) timestep into (*, timestep_embed_dim).

            Args:
                t: Tensor with shape (..., 1)
                timestep_embed_dim: embedding dimension (must be divisible by 2)

            Returns:
                Tensor with shape (..., timestep_embed_dim)
            """
            assert t.shape[-1] == 1
            assert timestep_embed_dim % 2 == 0

            device = t.device
            dtype = t.dtype

            freqs = 2 ** torch.arange(timestep_embed_dim // 2, device=device, dtype=dtype)
            scaled_t = t * freqs

            out = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)

            assert out.shape == (*t.shape[:-1], timestep_embed_dim)
            return out
        
    def forward(self, obs, action, time):
        #obs_emb = self.obs_embedding(obs)
        #action_emb = self.action_embedding(action)
        #time_emb = self.time_embedding(time)
        
        #x = torch.cat([obs_emb, action_emb, time_emb], dim=-1)
        time_embed = self.embed_timestep(time, 8)
        x = torch.cat([obs, action, time_embed], dim=-1)
        x = self.layer(x)
        step:ValueStep = self.head(x)
        
        return step
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, hidden_dims:list[int], norm_position:NormPosition=NormPosition.POST):
        super().__init__()

        #if norm is set true, the model will adapt layer norm
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(), norm_position=norm_position)

        self.head = CriticHead(feature_dim)

    def forward(self, x:torch.Tensor) -> ValueStep:
        x = self.layers(x)

        step:ValueStep = self.head(x)

        return step