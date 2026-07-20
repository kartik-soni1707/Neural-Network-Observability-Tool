from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)                    # FIX 1: list of tokens, not set
        res = []

        def count_freq():
            mapper = {}
            for k in range(len(tokens) - 1):     # FIX 2: adjacent tokens in the list,
                i, j = tokens[k], tokens[k + 1]  # not substring-scanning the string
                key = f"{i}@{j}"
                if key not in mapper:
                    mapper[key] = 0
                mapper[key] += 1
            return mapper

        def get_most_freq(dictionary):
            max_v = 0
            ans = []
            for k, v in dictionary.items():
                if v > max_v:
                    max_v = v                    # FIX 3: actually update max_v
                    ans = [k]
                elif v == max_v:
                    ans.append(k)
            ans.sort()
            return ans[0]

        def do_merge(v1, v2):                    # FIX 4: the missing step —
            new_tokens = []                      # rebuild the sequence after each merge
            k = 0
            while k < len(tokens):
                if k < len(tokens) - 1 and tokens[k] == v1 and tokens[k + 1] == v2:
                    new_tokens.append(v1 + v2)
                    k += 2
                else:
                    new_tokens.append(tokens[k])
                    k += 1
            return new_tokens

        for _ in range(num_merges):
            freqs = count_freq()
            if not freqs:
                break
            v1, v2 = get_most_freq(freqs).split("@")
            res.append([v1, v2])
            tokens = do_merge(v1, v2)            # FIX 1 again: sequence evolves
        return res