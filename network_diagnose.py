import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np
from math import exp

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        res = []
        inpt = x
        with torch.no_grad():
            for layer in model:
                inpt = layer(inpt)
                if isinstance(layer, nn.Linear):
                    res.append({
                        'mean': round(inpt.mean().item(), 4),
                        'std': round(inpt.std().item(), 4),
                        
                    })
                    if inpt.dim() >= 2:
                        res[-1]['dead_fraction'] = round(((inpt <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        res[-1]['dead_fraction'] = round((inpt <= 0).float().mean().item(), 4)

        return res

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        yhat = model(x)                 # simpler than looping layers manually
        loss = nn.MSELoss()(yhat, y)
        loss.backward()

        res = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                g = layer.weight.grad
                res.append({
                    'mean': round(g.mean().item(), 4),
                    'std':  round(g.std().item(), 4),
                    'norm': round(g.norm().item(), 4),
                })
        return res
    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i,x in enumerate(zip(activation_stats,gradient_stats)):
            activation_stat, gradient_stat=x 
            if activation_stat['dead_fraction']>0.5:
                return 'dead_neurons'
            if gradient_stat['norm']>1000:
                return 'exploding_gradients'
            if i==0 and gradient_stat['norm']<exp(-5):
                return 'vanishing_gradients'
            if activation_stat['std']<0.1:
                return 'vanishing_gradients'
            if activation_stat['std']>10:
                return 'exploding_gradients'
        return 'healthy'
