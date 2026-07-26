## Context

We have a key‑value store that keeps the last **T** tokens of a transformer model.  
Each value is a vector of dimensionality **d**.  
In the uncompressed form each token is stored as fp16 (2 bytes per element).  

A compression scheme stores most of the KV table in int4, but keeps a residual
window of the most recent **R** tokens in full precision to avoid catastrophic
error accumulation.  Additionally, the quantized values are grouped into blocks
of size **G**; each block has a scale factor (fp16) and a zero‑point (int4).
The memory cost is therefore

$$
\text{bytes}_{\text{quant}} =
\frac{\text{nbits}}{8}\,(T-R)\,d \;+\;
2\,R\,d \;+\;
G_{\text{cnt}}\,(s_{\text{size}}+z_{\text{size}})
$$

where  

* $G_{\text{cnt}}=\lceil (T-R)/G\rceil$ is the number of groups,
* $s_{\text{size}}=2$ bytes for a fp16 scale, and
* $z_{\text{size}}=1$ byte for two packed int4 zero‑points.

The full cache uses

$$
\text{bytes}_{\text{full}} = 2\,T\,d .
$$

The compression ratio is $\displaystyle \frac{\text{bytes}_{\text{full}}}{\text{bytes}_{\text{quant}}}$.

## Task

Implement the function `kv_memory_usage` that returns the total number of bytes
required to store a compressed KV table with the above layout.

```python
def kv_memory_usage(T: int, d: int, nbits: int = 4,
                    R: int = 0, group_size: int = 1) -> int:
    ...
```

* `T` – total number of tokens in the cache.  
* `d` – dimensionality of each value vector.  
* `nbits` – bits per quantized element (default 4).  
* `R` – size of the residual window that stays in fp16.  
* `group_size` – number of tokens per scale/zero‑point group.

The function must return an **integer** number of bytes.  All arithmetic should
be performed with Python integers; no NumPy is required.

## Example

```python
>>> kv_memory_usage(T=10, d=4, nbits=4, R=2, group_size=3)
41
```

The returned value equals

$$\frac{4}{8}\,(10-2)\,4 \;+\; 2\,2\,4 \;+\;
\left\lceil\frac{10-2}{3}\right\rceil\,(2+1) =
16 + 16 + 9 = 41 \text{ bytes}. $$

## What the gate checks

The grader computes a reference implementation using NumPy for the arithmetic
and compares the compression ratio produced by your function to that of the
oracle.  The solution must match the oracle within a relative tolerance of
$10^{-9}$.  A wrong formula will produce a different ratio and fail the gate.
