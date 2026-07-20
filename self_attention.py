import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        self.d_model=attention_dim
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Wk=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.Wq=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.Wv=nn.Linear(embedding_dim,attention_dim,bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.Wk(embedded)                                  # (B, T, A)
        Q = self.Wq(embedded)
        V = self.Wv(embedded)

        # 2. scores — transpose LAST TWO dims only (.T reverses ALL dims,
        #    shoving batch into the matmul: the error you hit before)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)   # (B, T, T)

        # 3. causal mask: lower-triangular T x T; 0s above the diagonal
        #    mark "future" positions -> set their scores to -inf
        T = embedded.shape[1]
        mask = torch.tril(torch.ones(T, T))                    # (T, T), broadcasts over B
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. softmax over the last dim: each row = one token's attention budget
        weights = torch.softmax(scores, dim=2)

        # 5. weighted blend of values
        return torch.round(weights @ V, decimals=4)            # (B, T, A)
