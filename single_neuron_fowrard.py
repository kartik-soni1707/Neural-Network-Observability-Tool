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

# Single neuron works by activation (w*x+b)
if __name__ == "__main__":
    sol = Solution()

    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, -0.5, 1.0])
    b = 0.5   # z = 0.5 - 1.0 + 3.0 + 0.5 = 3.0

    print(sol.forward(x, w, b, "sigmoid"))   # 0.95257
    print(sol.forward(x, w, b, "relu"))      # 3.0

    b2 = -4.0  # z = 2.5 - 4.0 = -1.5
    print(sol.forward(x, w, b2, "sigmoid"))  # 0.18243
    print(sol.forward(x, w, b2, "relu"))     # 0.0

