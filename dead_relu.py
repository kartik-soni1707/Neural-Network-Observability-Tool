import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        res=[]
        for layer in model:
            x=layer(x)
            m,n=x.shape
            if isinstance(layer, nn.ReLU):
                empty_cols = (x == 0).all(dim=0)      # bool tensor, one entry per column
                num_empty = empty_cols.sum().item()   # count
                res.append(round(num_empty/n,4))

        return res

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        max_dead=0
        prev=0
        increasing=True
        for i,fraction in enumerate(dead_fractions):
            max_dead=max(max_dead,fraction)
            if fraction<=prev:
                increasing=False
            if fraction >0.5:
                return 'use_leaky_relu'
            if i==0 and fraction>0.3:
                return 'reinitialize'
            if i==len(dead_fractions)-1 and increasing and fraction>0.1:
                return 'reduce_learning_rate'
            prev=fraction
        return 'healthy'
        
# Dead relu neurons can cause issues and common fixes
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 10)
    sol = Solution()

    # normal init → should be mostly healthy
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                          nn.Linear(32, 32), nn.ReLU(),
                          nn.Linear(32, 4))
    fr = sol.detect_dead_neurons(model, x)
    print("healthy-ish:", fr, "->", sol.suggest_fix(fr))

    # murder a layer: huge negative bias → all pre-activations < 0
    dead_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                               nn.Linear(32, 32), nn.ReLU(),
                               nn.Linear(32, 4))
    with torch.no_grad():
        dead_model[2].bias.fill_(-100.0)      # second Linear → its ReLU goes fully dead
    fr = sol.detect_dead_neurons(dead_model, x)
    print("killed layer:", fr, "->", sol.suggest_fix(fr))   # expect use_leaky_relu

    # pure-logic checks on suggest_fix
    print(sol.suggest_fix([0.4, 0.1]))          # reinitialize (first layer > 0.3)
    print(sol.suggest_fix([0.05, 0.12, 0.2]))   # reduce_learning_rate (increasing, last > 0.1)
    print(sol.suggest_fix([0.02, 0.05]))        # healthy