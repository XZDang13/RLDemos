import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, CategoricalHead, GuassianHead, CriticHead

class CategoricalActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        
        self.encoder, dim = make_mlp_layers(obs_dim, hidden_dims, F.silu, True)
        self.policy = CategoricalHead(dim, action_dim)
        
    def forward(self, obs, action=None):
        obs = self.encoder(obs)
        
        pi, action, log_prob = self.policy(obs, action)
        
        return pi, action, log_prob
    
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        
        self.encoder, dim = make_mlp_layers(obs_dim, hidden_dims, F.silu, True)
        self.policy = GuassianHead(dim, action_dim)
        
    def forward(self, obs, action=None):
        obs = self.encoder(obs)
        
        pi, action, log_prob = self.policy(obs, action)
        
        return pi, action, log_prob
    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super().__init__()
        
        self.encoder, dim = make_mlp_layers(obs_dim, hidden_dims, F.silu, True)
        self.value = CriticHead(dim)
        
    def forward(self, obs):
        obs = self.encoder(obs)
        
        value = self.value(obs)
        
        return value