## Context

The arithmetic intensity of a kernel is the ratio of floating‑point operations to bytes moved from memory.  
For a matrix multiplication \(E \in \mathbb{R}^{S\times d}\) times \(W \in \mathbb{R}^{d\times d}\) that produces an output \(O\), the number of multiply–add pairs is

$$
\mathrm{FLOPs}=2\,S\,d^{2}.
$$

The memory traffic consists of reading \(E\) and \(W\) once each, and writing \(O\) once:

$$
\mathrm{Bytes}=(S d + d^{2} + S d)\,\mathrm{bsize}
          =(2S d + d^{2})\,\mathrm{bsize},
$$

where \(\mathrm{bsize}\) is the size in bytes of a single element (e.g. 4 for `float32`, 8 for `float64`).  
Thus the arithmetic intensity is

$$
\mathrm{AI}= \frac{2S d^{2}}{(2S d + d^{2})\,\mathrm{bsize}}
          = \frac{2S d}{(2S + d)\,\mathrm{bsize}}.
$$

This closed‑form expression allows a quick comparison of the prefill phase against the decode phase in large language models.

## Task

Implement `prefill_arith_intensity(S, d, dtype='float32')`:

```python
def prefill_arith_intensity(S: int, d: int, dtype: str = 'float32') -> float:
    ...
```

The function receives the sequence length \(S\), hidden dimension \(d\), and a NumPy dtype name (`'float32'`, `'float64'`, …).  
It must return the arithmetic intensity as a `float` (Python `float`, i.e. double precision) computed from the closed‑form formula above.

## Example

```python
import numpy as np
ai = prefill_arith_intensity(128, 768, 'float32')
print(ai)
# ≈ 0.0009765625   # 2*128*768 / ((2*128+768)*4)
```

## What the gate checks

The grader evaluates the function on a set of random \((S,d)\) pairs and dtypes.  
It recomputes the reference value using NumPy’s `dtype.itemsize` to obtain the element size, then compares the candidate result with a relative tolerance of \(10^{-12}\).  The gate passes only if all comparisons succeed.
