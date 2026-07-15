import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        u = np.mean(x)
        std = (np.var(x) + 1e-5) ** 0.5
        out = gamma * (x - u) / std + beta
        return np.round(out, 5)

#Layer normalization normalizes across features within each sample, making it independent of batch size and ideal for transformers.
if __name__ == "__main__":
    x     = np.array([2.0, 4.0, 6.0, 8.0])
    gamma = np.array([1.0, 1.0, 2.0, 0.5])
    beta  = np.array([0.0, 0.1, -1.0, 3.0])

    print(Solution().forward(x, gamma, beta))

    plain = Solution().forward(x, np.ones_like(x), np.zeros_like(x))
    print("mean:", plain.mean().round(5), "std:", plain.std().round(5))