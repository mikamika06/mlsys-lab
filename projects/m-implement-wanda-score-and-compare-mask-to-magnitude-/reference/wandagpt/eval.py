from typing import List, Dict
import numpy as np
from wandagpt.score import wanda_score, create_mask_from_score, magnitude_mask
from wandagpt.sparsegpt import simplified_sparsegpt


class ToyLM:
    def __init__(self, weights: List[np.ndarray]):
        self.weights = [w.copy() for w in weights]

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for W in self.weights:
            h = np.maximum(0, h @ W)
        return h


def eval_perplexity(model, dataloader) -> float:
    """Compute mean cross-entropy loss converted to perplexity."""
    total_loss = 0.0
    total_count = 0

    for X, Y_targets in dataloader:
        logits = model.forward(X)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        batch_size = X.shape[0]
        correct_probs = probs[np.arange(batch_size), Y_targets]
        loss = -np.log(np.maximum(correct_probs, 1e-10))

        total_loss += np.sum(loss)
        total_count += batch_size

    avg_loss = total_loss / max(1, total_count)
    return float(np.exp(avg_loss))


def prune_and_eval_curve(
    model, dataloader, sparsities: List[float], method: str
) -> Dict[float, float]:
    """Evaluate perplexity curve across given sparsity values."""
    X_all = np.vstack([x for x, _ in dataloader])

    results = {}
    for sp in sparsities:
        pruned_weights = []
        h = X_all
        for W in model.weights:
            if method == "magnitude":
                m = magnitude_mask(W, sp)
                W_p = W * m
            elif method == "wanda":
                s = wanda_score(W.T, h.T)
                m = create_mask_from_score(s, sp).T
                W_p = W * m
            elif method == "sparsegpt":
                W_p = simplified_sparsegpt(W.T, h.T, sp).T
            else:
                raise ValueError(f"Unknown method {method}")

            pruned_weights.append(W_p)
            h = np.maximum(0, h @ W_p)

        temp_model = ToyLM(pruned_weights)
        ppl = eval_perplexity(temp_model, dataloader)
        results[sp] = ppl

    return results
