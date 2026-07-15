import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # --- forward ---
        z = np.dot(x, w) + b              # cleaner than the manual loop
        y_hat = 1 / (1 + np.exp(-z))      # sigmoid

        # --- backward (chain rule) ---
        dL_dyhat = y_hat - y_true         # dL/dy_hat
        dyhat_dz = y_hat * (1 - y_hat)    # sigmoid derivative
        dz = dL_dyhat * dyhat_dz          # dL/dz  (a scalar)

        dL_dw = dz * x                    # dz/dw = x  -> elementwise, gives a vector
        dL_db = dz                        # dz/db = 1  -> scalar

        return np.round(dL_dw, 5), round(float(dL_db), 5)

#We update weights by propagating error backwards 
#Use chain rule to compute gradients 
# --- setup ---
np.random.seed(0)
x = np.array([0.5, -1.2, 2.0])
w = np.array([0.1, 0.4, -0.3])
b = 0.2
y_true = 1.0

sol = Solution()
dw, db = sol.backward(x, w, b, y_true)
print("analytical dL/dw:", dw)
print("analytical dL/db:", db)