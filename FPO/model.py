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
        
        self.obs_embedding = MLPLayer(obs_dim, embedding_dim, nn.SiLU(), norm_position)
        self.action_embedding = MLPLayer(action_dim, embedding_dim, nn.SiLU(), norm_position)
        self.time_embedding = MLPLayer(1, embedding_dim, nn.SiLU(), norm_position)
        
        self.layer, feature_dim = make_mlp_layers(embedding_dim*3, hidden_dims, nn.SiLU(), norm_position)
        
        self.head = DiffusionHead(feature_dim, action_dim, velocity_scale)
        
        
    def forward(self, obs, action, time):
        obs_emb = self.obs_embedding(obs)
        action_emb = self.action_embedding(action)
        time_emb = self.time_embedding(time)
        
        x = torch.cat([obs_emb, action_emb, time_emb], dim=-1)
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