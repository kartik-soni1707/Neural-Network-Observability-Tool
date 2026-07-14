import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0) 
        self.embed = nn.Embedding(vocabulary_size, 16)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res=[]                  # x: (B, T) int token ids
        for sentence in x:
            op=self.embed(sentence[0])
            for word in sentence[1:]:
                op+=self.embed(word)
            
            op/=len(sentence)
            res.append(self.sigmoid(self.fc(op)))
        return torch.round(torch.stack(res), decimals=4)   # stacks list of (1,) into (B, 1)
