from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        def fun(no):
            if not no:
                return []                          # FIX 4: always return a list
            r = 0
            while r < len(no) and no[:r+1] in vocab:   # FIX 1: check keys, on the string
                r += 1
            if r == 0:
                r = 1                              # FIX 3: consume at least 1 char (or raise)
            ans = no[:r]                           # FIX 1: the token IS the string — no reverse lookup
            return [ans] + fun(no[r:])             # FIX 2: flat list, single recursive call

        res = []
        for i in numbers:
            res.append(fun(str(i)))
        return res

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        count = 0
        p1 = 0
        while p1 < len(text):
            # find the longest prefix starting at p1 that IS in vocab
            best = p1 + 1                       # consume at least 1 char (fallback)
            p2 = p1 + 1
            while p2 <= len(text):
                if text[p1:p2] in vocab:
                    best = p2                   # remember this valid match
                p2 += 1
            count += 1
            p1 = best                           # jump to end of longest valid match
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words=len(text.split(" "))
        tokens=self.count_tokens(text,vocab)
        return round(tokens/words,4)