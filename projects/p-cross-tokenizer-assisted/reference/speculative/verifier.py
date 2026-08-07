import numpy as np

def verify_distribution(target_probs, draft_probs):
    kl = np.sum(target_probs * np.log((target_probs + 1e-10) / (draft_probs + 1e-10)))
    return kl < 2.0
