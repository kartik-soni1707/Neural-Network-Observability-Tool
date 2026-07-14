import numpy as np
from numpy.typing import NDArray
from math import sin,cos

class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        res = []
        for pos in range(seq_len):
            row = []
            for i in range(d_model // 2):
                angle = pos / (10000 ** (2 * i / d_model))
                row.append(round(sin(angle), 5))
                row.append(round(cos(angle), 5))
            res.append(row)
        return np.array(res)