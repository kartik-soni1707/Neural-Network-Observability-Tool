import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b1 = np.array(b1, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ---- Forward ----
        z1 = W1 @ x + b1                  # (hidden,) pre-activation
        out1 = np.maximum(z1, 0)          # (hidden,) post-ReLU
        y_pred = W2 @ out1 + b2           # (out,)

        loss = ((y_pred - y_true) ** 2)[0]
        

        # ---- Backward ----
        
        grad_y = (2.0) * (y_pred - y_true)      # (out,)  dL/dy_pred

        dW2 = np.outer(grad_y, out1)                 # (out, hidden)
        db2 = grad_y                                 # (out,)

        grad_hidden = W2.T @ grad_y                  # (hidden,) back through W2
        grad_z1 = grad_hidden * (z1 > 0)             # ReLU mask
        dW1 = np.outer(grad_z1, x)                   # (hidden, input)
        db1 = grad_z1                                # (hidden,)

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }