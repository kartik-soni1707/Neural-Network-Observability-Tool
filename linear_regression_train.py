import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights=initial_weights
        for _ in range(num_iterations):
            prediction=self.get_model_prediction(X,weights)
            for j in range(len(weights)):
                weights[j]-=self.learning_rate*self.get_derivative(prediction,Y,len(X),X,j)
        for j in range(len(weights)):
            weights[j]=round(weights[j],5)
        return weights
# Fundamental/ unit learning neuron
if __name__ == "__main__":
    sol = Solution()

    # Data generated from y = 2*x1 + 3*x2 (no noise), so training should
    # recover weights close to [2, 3]
    X = np.array([[1.0, 1.0],
                  [2.0, 1.0],
                  [1.0, 2.0],
                  [3.0, 2.0],
                  [2.0, 3.0]])
    Y = np.array([5.0, 7.0, 8.0, 12.0, 13.0])

    initial_weights = np.array([0.0, 0.0])
    final = sol.train_model(X, Y, num_iterations=1000, initial_weights=initial_weights)
    print(final)                                   # ~[2. 3.]
    print(sol.get_model_prediction(X, final))      # ~[5. 7. 8. 12. 13.]