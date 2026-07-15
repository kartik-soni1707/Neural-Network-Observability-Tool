import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        m, n = X.shape            # m samples, n features
        w = np.zeros(n)           # shape (n,)
        b = 0.0

        for _ in range(epochs):
            yhat = X @ w + b                  # (m,)
            err = yhat - y                    # (m,)
            dw = (2 / m) * (X.T @ err)        # (n,)
            db = (2 / m) * np.sum(err)        # scalar
            w -= lr * dw
            b -= lr * db

        return np.round(w, 5), round(b, 5)

#The base for all forward, loss, backprop, update weights
if __name__ == "__main__":
    # synthetic data with KNOWN answer: y = 3x1 - 2x2 + 5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    true_w, true_b = np.array([3.0, -2.0]), 5.0
    y = X @ true_w + true_b + rng.normal(scale=0.01, size=200)  # tiny noise

    w, b = Solution().train(X, y, epochs=1000, lr=0.1)
    print("learned w:", w, "  b:", b)      # expect ≈ [3, -2], 5

    # watch convergence: fewer epochs = less recovered
    for e in [10, 100, 1000]:
        w_e, b_e = Solution().train(X, y, epochs=e, lr=0.1)
        loss = np.mean((X @ w_e + b_e - y) ** 2)
        print(f"epochs={e:5d}  loss={loss:.6f}  w={w_e}")