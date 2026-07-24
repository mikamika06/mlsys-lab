## Context

In many language‑model compression schemes a *compression ratio* $r \in [0,1]$ is used to indicate the fraction of key–value (KV) memory that can be discarded while still retaining enough tokens for inference.  
If a sequence contains $n$ tokens and the full KV buffer occupies $B$ bytes, then:

- The number of **kept** tokens is
  $$k = \operatorname{round}\!\bigl((1-r)\, n\bigr).$$

- The amount of **memory saved** is simply the discarded fraction:
  $$s = r\, B.$$

The task is to implement a function that returns these two values.

## Task

Implement `measure_kept_tokens_and_memory`:

```python
def measure_kept_tokens_and_memory(compression_ratio: float,
                                   seq_len: int,
                                   full_bytes: int) -> tuple[int, float]:
    ...
```

- `compression_ratio` is a float in $[0,1]$.
- `seq_len` is the number of tokens (positive integer).
- `full_bytes` is the size of the KV buffer before compression (positive integer).

The function must return a tuple `(kept_tokens, memory_saved)` where:

* `kept_tokens` is an **integer** equal to $\operatorname{round}((1-r)\,n)$.
* `memory_saved` is a **float64** equal to $r\,B$.

Use NumPy for the arithmetic; do not hard‑code any values.

## Example

```python
import numpy as np
kept, saved = measure_kept_tokens_and_memory(0.25, 100, 8000)
print(kept)   # 75
print(saved)  # 2000.0
```

Here $k=\operatorname{round}(0.75\times100)=75$ and $s=0.25\times8000=2000$.

## What the gate checks

The grader computes a reference implementation using NumPy:

```python
import numpy as np
ref_kept = int(np.round((1 - r) * n))
ref_saved = (r * B)
```

It then verifies that:

* `kept_tokens` matches `ref_kept` exactly.
* The relative error of `memory_saved` satisfies  
  $$\frac{|\,\text{student} - \text{reference}\,|}{|\text{reference}| + 10^{-12}} \le 10^{-9}.$$

If both conditions hold the solution passes; otherwise it fails.
