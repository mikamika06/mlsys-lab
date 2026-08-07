import math

def h2o_eviction_set(attn_scores: list[list[float]], budget: int, recent_window: int):
    """
    H2O (Heavy-Hitter Oracle) static eviction set.

    attn_scores : (n, n) raw attention logits.
    budget      : number of tokens to keep (recent_window <= budget <= n).
    recent_window : number of most-recent positions always kept.

    Returns (retained_idx, preserved_mass):
      retained_idx   : 1-D int64 array, ascending, length == budget.
      preserved_mass : float, fraction of total accumulated attention mass
                        captured by the retained set.
    """
    raise NotImplementedError('your code here')
