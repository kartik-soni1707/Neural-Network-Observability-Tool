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
# Preserve signal by averaging over the columns in batch
#Good for CNNs etc large batch and fixed cols

if __name__ == "__main__":
    x = [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]          # batch of 3 samples, 2 features
    gamma = [1.0, 2.0]
    beta  = [0.0, 0.5]
    rm, rv = [0.0, 0.0], [1.0, 1.0]
    momentum, eps = 0.1, 1e-5

    sol = Solution()

    # training step: uses batch stats, updates running stats
    y, rm, rv = sol.batch_norm(x, gamma, beta, rm, rv, momentum, eps, training=True)
    print("train y:", y)
    print("running_mean:", rm)     # nudged 10% toward batch mean [3, 4]
    print("running_var: ", rv)

    # inference: uses running stats, batch of ONE works fine
    y2, rm2, rv2 = sol.batch_norm([[2.0, 3.0]], gamma, beta, rm, rv, momentum, eps, training=False)
    print("infer y:", y2)
    print("stats unchanged:", rm2 == rm, rv2 == rv)