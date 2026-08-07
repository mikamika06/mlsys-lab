import numpy as np

def transfer_candidates(draft_candidates, target_vocab):
    mapped = []
    for c in draft_candidates:
        if c in target_vocab:
            mapped.append(target_vocab[c])
        else:
            mapped.append(0)
    return mapped
