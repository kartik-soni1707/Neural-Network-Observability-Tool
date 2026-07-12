import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        # rms=0
        # for i in x:
        #     rms+=i**2
        # rms/=len(x)
        # rms+=eps
        # rms=rms**0.5
        # for i in range(len(x)):
        #     x[i]=np.round((x[i]/rms)*gamma[i],decimals=4)
        #return x
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        rms = np.sqrt(np.mean(x**2) + eps)
        return np.round(gamma * x / rms, 4).tolist()

