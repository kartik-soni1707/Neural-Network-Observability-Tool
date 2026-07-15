import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        res=[]
        for x in X:
            val=0
            for k,v in zip(x,weights):
                val+=k*v
                
            res.append(round(val,5))
        return res

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        res=0
        for x,y in zip(model_prediction,ground_truth):
            res+=(x-y)**2
        res/=len(model_prediction)
        res=res[0]
        return round(res,5)
#Kind of atomic operation or the logic per layer
if __name__ == "__main__":
    sol = Solution()

    # 3 samples, 2 features
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    weights = np.array([0.5, -1.0])

    preds = sol.get_model_prediction(X, weights)
    print(preds)   # [-1.5, -2.5, -3.5]

    ground_truth = np.array([[-1.0], [-2.0], [-4.0]])
    print(sol.get_error(np.array(preds), ground_truth))   # 0.25