import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        beta = np.array(beta, dtype=np.float64)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)
        if training:
            u=np.mean(x,axis=0)
            var=np.var(x,axis=0)
            std=(var+eps)**(0.5)
            xhat=(x-u)/std
    
            running_mean= np.round((1-momentum)*running_mean + momentum* u,decimals=4)
            running_var= np.round((1-momentum)*running_var + momentum* var,decimals=4)
        
        else:
            xhat=(x-running_mean)/(running_var+eps)**0.5
        
        y=np.round((gamma*xhat+beta),decimals=4)
        return (y, running_mean, running_var)
