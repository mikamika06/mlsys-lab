import math

def h2o_eviction_trajectory(K: list[list[float]], Q: list[list[float]], prompt_len: int, budget: int, recent_window: int) -> list[list[int]]:
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
    raise NotImplementedError('your code here')
