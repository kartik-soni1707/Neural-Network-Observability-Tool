import numpy as np
from numpy.typing import NDArray
from math import log

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        res=0
        def helper(true,pred):
            val=0
            val= true*log(pred)+(1-true)*log(1-pred)
            return val
        for x,y in zip(y_true,y_pred):
            res-=helper(x,y)
        res/=len(y_pred)
        return round(res,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        res=0
        def helper(true,pred):
            val=0
            for t,p in zip(true,pred):
                val+=t*log(p)
            return val
        for x,y in zip(y_true,y_pred):
            res-=helper(x,y)
        res/=len(y_pred)
        return round(res,4)

#Loss function for classification problems 
if __name__ == "__main__":
    sol = Solution()

    # Binary cross-entropy
    y_true = np.array([1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7])
    print(sol.binary_cross_entropy(y_true, y_pred))   # 0.2027

    # Categorical cross-entropy (3 samples, 3 classes)
    y_true_c = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
    y_pred_c = np.array([[0.7, 0.2, 0.1],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.2, 0.6]])
    print(sol.categorical_cross_entropy(y_true_c, y_pred_c))   # 0.3635