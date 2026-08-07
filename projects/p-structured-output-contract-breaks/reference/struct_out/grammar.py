import numpy as np

class GrammarProcessor:
    def process(self, logits, mask):
        arr = np.array(logits, dtype=float)
        m = np.array(mask, dtype=bool)
        arr[~m] = -1e9
        return arr
