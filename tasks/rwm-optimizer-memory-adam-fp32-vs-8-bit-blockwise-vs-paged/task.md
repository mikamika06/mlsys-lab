## Context

Adam optimizer maintains two state tensors per parameter: the first‑moment estimate $m$ and the second‑moment estimate $v$. In a typical FP32 implementation each of these is stored as a 32‑bit floating point number. For a model with $\Phi$ parameters the memory cost is therefore

$$
M_{\text{FP32}} = \Phi \times 2 \times 4\;\text{bytes} = 8\,\Phi.
$$

A popular compression strategy stores $m$ and $v$ as 8‑bit integers. To recover a good approximation one keeps, for each contiguous block of size $B$, the maximum absolute value $\alpha$ that appears in that block; all values are then scaled by $\alpha$. The memory cost becomes

$$
M_{\text{Block}} = \Phi \times 2 \times 1\;\text{bytes} + 2 \times \Bigl\lceil \frac{\Phi}{B}\Bigr\rceil \times 4\;\text{bytes},
$$

where the second term accounts for the two $\alpha$ scalars (one per state) stored as 32‑bit floats.

A paged variant eliminates the steady overhead of storing the block maxima by keeping them only when a page is resident. In this simplified model we assume that the paging mechanism does not add any extra bytes, so

$$
M_{\text{Paged}} = \Phi \times 2 \;\text{bytes}.
$$

## Task

Implement `estimate_memory(num_params: int, block_size: int) -> Tuple[int, int, int]` that returns a tuple `(fp32_bytes, blockwise_bytes, paged_bytes)` containing the exact memory cost in bytes for each representation described above.

The function must:

1. Accept only non‑negative integers.
2. Compute the ceiling division correctly when `block_size` does not divide `num_params`.
3. Return integer byte counts (no floating point).

## Example

```python
>>> estimate_memory(1000, 128)
(8000, 2064, 2000)
```

The first value is $8\times1000=8000$ bytes for FP32.  
For blockwise: $2\times1000 + 2\times\lceil1000/128\rceil\times4 = 2000 + 2\times8\times4 = 2064$.  
Paged uses only the two 1‑byte states per parameter, so $2\times1000=2000$.

## What the gate checks

The grader generates several random test cases and compares your output to a reference computed by the same formulas. The comparison is exact: the tuple must match exactly for all three values. No tolerance is applied because the numbers are integers.
