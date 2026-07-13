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