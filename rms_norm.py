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

if __name__ == "__main__":
    x     = [2.0, -4.0, 6.0, -8.0]
    gamma = [1.0, 1.0, 0.5, 2.0]
    eps   = 1e-6

    out = Solution().rms_norm(x, gamma, eps)
    print("out:", out)

    # sanity: with gamma=1, the output's RMS should be ≈ 1
    plain = Solution().rms_norm(x, [1.0]*len(x), eps)
    print("rms of output:", round(float(np.sqrt(np.mean(np.array(plain)**2))), 5))

    # contrast with LayerNorm: mean is NOT forced to zero
    print("mean of output:", round(float(np.mean(plain)), 5))