## Context

In transformer‑style attention, the most memory‑intensive operation is the pairwise score matrix between tokens.  
For a sequence of length $N$, with $H$ heads and per‑head dimension $D$, the naïve implementation stores an $N\times N$ matrix for each head:

$$
\text{naïve} = H \times N^2 \times s,
$$

where $s$ is the size in bytes of a single score (typically 8 bytes for `float64`).  
Flash‑style attention reorganises the computation so that only linear‑size tensors are kept.  
The dominant memory usage becomes the query, key and value tensors:

$$
\text{flash} = 3 \times H \times N \times D \times s.
$$

Because $N^2$ grows quadratically while $ND$ grows linearly, the gap between the two approaches widens rapidly as the sequence length increases.

## Task

Implement a function that returns the peak memory usage (in bytes) of both the naïve and flash attention mechanisms for given parameters:

```python
def peak_attention_memory(N: int, heads: int, d: int) -> tuple[int, int]:
    """
    Return (naive_bytes, flash_bytes).
    """
```

The function must use only integer arithmetic; no external libraries are required.

## Example

```python
>>> peak_attention_memory(4, 2, 3)
(256, 576)
```

Explanation:  
- Element size $s = 8$ bytes.  
- Naïve: $2 \times 4^2 \times 8 = 256$.  
- Flash: $3 \times 2 \times 4 \times 3 \times 8 = 576$.

## What the gate checks

The grader computes a reference implementation using the formulas above for several random test cases.  
Your function must return exactly the same tuple of integers; any mismatch causes the `exact_match` gate to fail.
