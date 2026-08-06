import math
import numpy as np


def h2o_eviction_trajectory(K: np.ndarray, Q: np.ndarray, prompt_len: int,
                             budget: int, recent_window: int) -> list[list[int]]:
    """Simulate H2O (Heavy-Hitter Oracle) KV-cache eviction over decode.

    K          : (prompt_len + T, d) keys for the prompt AND every token
                 that will be decoded, indexed by absolute position.
    Q          : (T, d) the query issued at each of T decode steps. Query
                 t attends over whatever is currently resident in the
                 cache (positions 0..prompt_len+t-1, minus anything
                 already evicted) BEFORE position prompt_len+t is
                 appended.
    prompt_len : number of prompt positions initially resident
                 (0..prompt_len-1). Assumed <= budget (no eviction needed
                 before decoding starts).
    budget     : maximum resident cache size.
    recent_window : number of most-recently-appended resident positions
                 that are always protected from eviction (>= 1).

    At each decode step t (0-indexed):
      1. Attend Q[t] over the currently resident positions (ascending
         order); softmax attention weights accumulate into each resident
         position's running heavy-hitter score.
      2. Append position `prompt_len + t` to the cache with score 0.
      3. If the cache now exceeds `budget`, evict ONE position: the
         lowest-scoring position among those NOT in the `recent_window`
         most-recently-appended resident positions (ties broken by
         smaller position index).

    Returns a list of length T; entry t is the sorted list of resident
    position indices immediately after step t.
    """
    K = np.asarray(K, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    d = K.shape[1]
    T = Q.shape[0]

    resident = list(range(prompt_len))
    score = {i: 0.0 for i in resident}

    trajectory: list[list[int]] = []
    for t in range(T):
        q = Q[t]
        idx = sorted(resident)
        Kc = K[idx]
        sqrt_d = math.sqrt(d)
        logits = [
            sum(q[k] * Kc[j][k] for k in range(d)) / sqrt_d
            for j in range(len(idx))
        ]
        max_logit = max(logits)
        logits = [l - max_logit for l in logits]
        w = [math.exp(l) for l in logits]
        sum_w = sum(w)
        w = [val / sum_w for val in w]
        for j, i in enumerate(idx):
            score[i] += float(w[j])

        new_pos = prompt_len + t
        resident.append(new_pos)
        score[new_pos] = 0.0

        if len(resident) > budget:
            current = sorted(resident)
            protected = set(current[-recent_window:]) if recent_window > 0 else set()
            evictable = [i for i in current if i not in protected]
            worst = min(evictable, key=lambda i: (score[i], i))
            resident.remove(worst)
            del score[worst]

        trajectory.append(sorted(resident))

    return trajectory
