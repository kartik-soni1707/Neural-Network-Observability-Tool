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

