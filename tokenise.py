import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words=set()
        for s1,s2 in zip(positive,negative):
            for w1 in s1.split(" "):
                words.add(w1)
            for w2 in s2.split(" "):
                words.add(w2)
        words=list(words)
        words.sort()
        op=[]
        max_l=0
        for sentence in positive:
            op.append([])
            for w in sentence.split(" "):
                op[-1].append(words.index(w)+1)
            max_l=max(max_l,len(op[-1]))
        for sentence in negative:
            op.append([])
            for w in sentence.split(" "):
                op[-1].append(words.index(w)+1)
            max_l=max(max_l,len(op[-1]))
        for i in range(len(op)):
            op[i]+=[0]*(max_l-len(op[i]))
        return op