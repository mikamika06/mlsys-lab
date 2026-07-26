## Context

In a streaming language model, the most recent tokens are kept in a sliding window while older tokens may be retained as *sink* for special purposes or discarded (*evicted*) to save memory. For a given current position $p$ in a sequence of length $n$, we consider all past indices $i < p$. The first $k$ indices ($0 \le i < k$) are designated **sink** tokens, the last $w$ indices before $p$ ($p-w \le i < p$) form the **window**, and every other past index is **evicted**.

We need a function that, for any $(k,w,p)$, returns an array of length $p$ with integer labels: $0$ for sink, $1$ for window, $2$ for evicted. The output must be a NumPy array of dtype `int64`.

## Task

Implement the following function:

```python
import numpy as np

def classify_past_tokens(k: int, w: int, pos: int) -> np.ndarray:
    """
    Return an array of length ``pos`` where each element is 0 (sink),
    1 (window), or 2 (evicted) according to the rules described in
    the context section.
    """
    ...
```

The function must be pure NumPy: no Python loops, no list comprehensions that iterate over indices. It should work for any non‑negative integers `k`, `w`, and `pos` with `0 <= k <= pos` and `0 <= w <= pos`.

## Example

```python
>>> import numpy as np
>>> classify_past_tokens(k=2, w=3, pos=7)
array([0, 0, 2, 2, 1, 1, 1])
```

Explanation: indices `0` and `1` are sink; indices `4`, `5`, `6` belong to the window; indices `2` and `3` are evicted.

## What the gate checks

The grader computes a reference array using NumPy logic for several test cases. Your output must match that reference exactly (`np.array_equal`). No other metrics are evaluated.
