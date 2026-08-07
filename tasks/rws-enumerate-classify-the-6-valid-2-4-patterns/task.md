## Context

In many binary‑feature settings we need to identify which two of four positions are active.  
There are $\binom{4}{2}=6$ distinct patterns that contain exactly two ones:
$$
\begin{array}{cccc}
0&0&1&1\\
0&1&0&1\\
0&1&1&0\\
1&0&0&1\\
1&0&1&0\\
1&1&0&0
\end{array}
$$
These six rows form a canonical table.  For any 4‑bit vector we must return the index of its row in this table, or ``-1`` if it does not contain exactly two ones.

## Task

Implement `classify_patterns(vectors)`:

```python
def classify_patterns(vectors: list[list[int]]) -> list[int]:
    ...
```

`vectors` is a 2‑D list of shape `(N,4)` with integer entries `0` or `1`.  
Return a 1‑D array of length `N` containing the pattern index (`0`–`5`) for each row that has exactly two ones, and `-1` otherwise. The result must be of type `int`.

## Example

```python
from classify_patterns import classify_patterns

A = [[0, 0, 1, 1],
              [1, 0, 0, 1],
              [1, 1, 1, 0]]   # last row has three ones
idxs = classify_patterns(A)
print(idxs)  # [0, 3, -1]
```

## What the gate checks

The grader builds a reference mapping from each valid 4‑bit vector to its canonical index and compares your output exactly. The metric `exact_match` is `1.0` only if every returned value matches the reference; otherwise it is `0.0`. No other performance or style constraints are enforced.
