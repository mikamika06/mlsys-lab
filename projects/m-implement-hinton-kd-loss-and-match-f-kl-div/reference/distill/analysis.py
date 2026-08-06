import torch
import torch.nn.functional as F


def softmax_entropy_curve(logits, temperatures):
    entropies = []
    for t in temperatures:
        probs = F.softmax(logits / t, dim=-1)
        log_probs = F.log_softmax(logits / t, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
        entropies.append(entropy)
    return entropies
