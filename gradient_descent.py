class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        res=init

        for i in range(iterations):
            res-=learning_rate*2*res
        return round(res,5)
#Learning point: stepping through loss function to find minima 
#learning_rate: single most important hyper parameter

if __name__ == "__main__":
    sol = Solution()
    print(sol.get_minimizer(0, 0.1, 5))    # 5.0
    print(sol.get_minimizer(10, 0.01, 5))  # 4.08536
    print(sol.get_minimizer(10, 0.1, 5))   # 0.53687
    print(sol.get_minimizer(10, 0.5, 5))   # 0.0