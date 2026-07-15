import numpy as np
from numpy.typing import NDArray
from math import exp

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z - np.max(z)
        exp_z = np.exp(z)
        tot = np.sum(exp_z)
        return np.round(exp_z / tot, 4)
    
#Convert logits too prob dist that add to 1
if __name__ == "__main__":
    sol = Solution()

    z1 = np.array([1.0, 2.0, 3.0])
    print(sol.softmax(z1))   # [0.09   0.2447 0.6652]

    z2 = np.array([1000.0, 1001.0, 1002.0])
    print(sol.softmax(z2))   # [0.09   0.2447 0.6652] — no overflow thanks to max subtraction

    z3 = np.array([5.0, 5.0, 5.0])
    print(sol.softmax(z3))   # [0.3333 0.3333 0.3333]