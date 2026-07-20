from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        text =list(set(text))
        text.sort()
        itos={}
        stoi={}
        for i in range(len(text)):
            itos[i]=text[i]
            stoi[text[i]]=i
        return (stoi,itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        res=[]
        for i in text:
            res.append(stoi[i])
        return res

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        res=[]
        for i in ids:
            res.append(itos[i])
        return "".join(res)
