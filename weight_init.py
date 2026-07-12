import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
    
        # Xavier/Glorot normal: std = sqrt(2 / (fan_in + fan_out))
        std = math.sqrt(2.0 / (fan_in + fan_out))
        
        # Create (fan_out x fan_in) matrix from normal dist with that std
        W = torch.randn(fan_out, fan_in) * std
        
        # Round to 4 decimals and convert to nested list
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
    
        # Xavier/Glorot normal: std = sqrt(2 / (fan_in + fan_out))
        std = math.sqrt(2.0 / (fan_in))
        
        # Create (fan_out x fan_in) matrix from normal dist with that std
        W = torch.randn(fan_out, fan_in) * std
        
        # Round to 4 decimals and convert to nested list
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> list[float]:
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
