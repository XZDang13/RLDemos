import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import DeterminicHead, CriticHead, make_mlp_layers
    
class DDPGActor(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(state_dim, hidden_dims, F.silu, True)
        self.max_action = max_action
        self.policy = DeterminicHead(dim, action_dim, max_action)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)

        action = self.policy(x)

        return action
    
class DDPGCritic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dims:list[int]):
        super().__init__()
        self.layers, in_dim = make_mlp_layers(state_dim+action_dim, hidden_dims, F.silu, True)
        self.critic_layer = CriticHead(in_dim)

    def forward(self, feature:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([feature, action], 1)
        x = self.layers(x)
        q = self.critic_layer(x)

        return q