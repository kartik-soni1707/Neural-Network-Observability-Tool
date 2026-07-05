import numpy as np
from numpy.typing import NDArray
from math import exp

class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        res=0
        for x,y in zip(x,w):
            res+=x*y
        res+=b
        if activation=="sigmoid":
            res = 1/(1+exp(-1*res))
        else:
            res=max(res,0.0)
        return round(res,5)

    

