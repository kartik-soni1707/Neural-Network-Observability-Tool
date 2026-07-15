import numpy as np
from numpy.typing import NDArray
from math import exp
class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        def helper(num):
            res=1/(1+exp(-1*num))
            return round(res,5)
        for i in range(len(z)):
            z[i]=helper(z[i])
        return z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        def helper(num):
            return max(0,num)
        for i in range(len(z)):
            z[i]=helper(z[i])
        return z

#Activation functions add non linearity
#Use sigmoid for binary output due to vanishing gradient
#Use relu for hidden layers does not lead to vanishing gradient 
if __name__ == "__main__":
    sol = Solution()

    z1 = np.array([0.0, 1.0, -1.0, 5.0])
    print(sol.sigmoid(z1))   # [0.5     0.73106 0.26894 0.99331]

    z2 = np.array([-2.0, 0.0, 3.5, -0.1])
    print(sol.relu(z2))      # [0.  0.  3.5 0. ]