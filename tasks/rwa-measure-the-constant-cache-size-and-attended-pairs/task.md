## Context

In many sequence‑to‑sequence models a fixed cache of size $k + w$ is maintained while processing an input of length $n$.  
For each position $i \in \{1,\dots,n\}$ the actual number of tokens that can be stored in the cache is therefore

$$
c_i = \min(k+w,\, i).
$$

The model also attends only to a sliding window of size $w$ around the current token.  
Consequently at step $i$ the number of *attended pairs* (pairs between the current token and tokens inside the window) equals

$$
p_i = \min(w,\; i-1).
$$

These two sequences $\{c_i\}$ and $\{p_i\}$ are deterministic functions of the hyper‑parameters $k$, $w$ and the sequence length $n$.

## Task

Implement a function

```python
def measure_cache_and_attended(k: int, w: int, seq_len: int) -> Tuple[List[int], List[int]]:
    ...
```

that returns two lists:

* `cache_sizes`: an integer list of length `seq_len` where `cache_sizes[i] = c_{i+1}`.
* `attended_pairs`: an integer list of length `seq_len` where `attended_pairs[i] = p_{i+1}`.

The function must use only plain Python and standard library types; no external libraries are required.

## Example

```python
>>> from your_module import measure_cache_and_attended
>>> cache, pairs = measure_cache_and_attended(k=2, w=3, seq_len=5)
>>> cache
[2, 3, 4, 5, 5]
>>> pairs
[0, 1, 2, 3, 3]
```

## What the gate checks

The grader computes a reference implementation (the oracle) for several test cases and compares the returned lists element‑wise.  
It also verifies that no reported cache size ever exceeds `k + w`.  The candidate solution must match the oracle exactly; otherwise the `exact_match` metric will be set to `0.0`.
