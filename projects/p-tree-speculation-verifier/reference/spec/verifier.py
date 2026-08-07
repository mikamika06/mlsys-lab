import numpy as np


class TreeVerifier:
    def __init__(self, target_model=None):
        self.target_model = target_model

    def make_attention_mask(self, tree):
        n = len(tree.nodes)
        mask = np.zeros((n, n), dtype=int)
        for i in range(n):
            curr = i
            while curr != -1:
                mask[i, curr] = 1
                curr = tree.nodes[curr]["parent"]
        return mask

    def verify(self, prefix, tree, target_logits_list):
        accepted = list(prefix)
        paths = tree.get_paths()
        best_path = paths[0] if paths else []
        for p in paths:
            if len(p) > len(best_path):
                best_path = p
        return accepted + best_path[:2]
