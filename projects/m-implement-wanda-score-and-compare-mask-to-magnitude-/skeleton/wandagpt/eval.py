from typing import List, Dict
import numpy as np


def eval_perplexity(model, dataloader) -> float:
    """Compute perplexity of a simple toy LM on given dataloader."""
    raise NotImplementedError


def prune_and_eval_curve(
    model, dataloader, sparsities: List[float], method: str
) -> Dict[float, float]:
    """Evaluate perplexity over multiple sparsity levels using specified method."""
    raise NotImplementedError
