## Context

Production inference engines often store key-value (KV) cache entries in fixed-size physical blocks instead of one contiguous tensor. A block table maps the logical sequence order to the physical storage locations, allowing blocks to be moved or reused without changing the logical sequence.

Assume keys and values are stored as physical blocks:

$$K_{\mathrm{phys}}, V_{\mathrm{phys}} \in \mathbb{R}^{B \times S \times H},$$

where $B$ is the number of physical blocks, $S$ is the block size, and $H$ is the head dimension. A logical sequence is reconstructed using a block table

$$T \in \mathbb{N}^{L_b},$$

where $T_i$ gives the physical block index for logical block $i$.

The logical key cache is obtained by gathering blocks in table order:

$$K_{\mathrm{logical}} = \mathrm{reshape}([K_{\mathrm{phys}}[T_0], K_{\mathrm{phys}}[T_1], \dots], (-1,H)).$$

Given a query vector $q \in \mathbb{R}^{H}$, attention scores are

$$s_i = \frac{q^\top K_{\mathrm{logical},i}}{\sqrt{H}}.$$

The output is the weighted value sum:

$$o = \sum_i \mathrm{softmax}(s)_i V_{\mathrm{logical},i}.$$

The implementation must combine block gathering and attention computation to produce the same result as if the cache had always been contiguous.

## Task

Implement `gather_attention`:

```python
def gather_attention(k_phys: list[list[list[float]]], v_phys: list[list[list[float]]], block_table: list[int], q: list[float]) -> list[float]:
    ...
```

Inputs:
- `k_phys` has shape $(B,S,H)$ and stores keys in physical block order.
- `v_phys` has shape $(B,S,H)$ and stores values in the same physical block order.
- `block_table` has shape $(L_b)$ and contains physical block indices in logical order.
- `q` has shape $(H)$.

Return a vector of shape $(H)$ containing the attention output in `float64`.

The returned value must match the contiguous-cache attention result. Use Python operations for gathering and numerical computation.

## Example

```python

k_phys = [
    [[1., 0.], [0., 1.]],
    [[2., 0.], [0., 2.]]
]
v_phys = k_phys.copy()
block_table = [1, 0]
q = [1., 0.]

out = gather_attention(k_phys, v_phys, block_table, q)
```

The logical cache begins with physical block `1` and then physical block `0`, so attention is performed over the reordered sequence.

## What the gate checks

The gate creates shuffled physical KV blocks and a logical-to-physical block table. It computes the reference output by gathering the logical cache with a Python oracle and applying the attention equations.

The submission output is compared with the oracle using

$$\max_i |o_i-\hat{o}_i| \le 10^{-6}.$$

Implementations that skip the block-table reorder or attend over the physical layout will fail.
