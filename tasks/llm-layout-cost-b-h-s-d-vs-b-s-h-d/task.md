## Context

In a multi‑head attention module the query, key and value tensors are often stored in one of two layouts:

* **BHSD** – batch × heads × sequence length × head dimension  
* **BSHD** – batch × sequence length × heads × head dimension  

Both shapes contain the same number of elements, but their memory layout differs.  In a row‑major (C‑order) array the stride for each dimension is the product of the sizes of all dimensions that follow it.  For example, in BHSD the strides are

$$
\text{strides}_{BHSD} = \bigl[H\,S\,d,\; S\,d,\; d,\; 1\bigr].
$$

When iterating over the tensor in nested loops that match the logical order of dimensions (B,H,S,d for BHSD, B,S,H,d for BSHD), the number of cache‑line misses depends on how often the stride between successive elements exceeds one element.  A larger stride forces a new cache line to be fetched more frequently.

The **cost** we want to estimate is simply the number of distinct cache lines accessed during a full traversal in the natural dimension order, assuming an 8‑byte `float64` element and a 64‑byte cache line (i.e. 8 elements per line).

## Task

Implement `layout_cost(shape)`:

```python
def layout_cost(shape: tuple[int, int, int, int]) -> tuple[int, int]:
    ...
```

* `shape` is the BHSD shape `(B, H, S, d)`.  
* The function must return a pair of integers:
  * `cost_bh`: number of cache lines accessed when the tensor is stored in **BHSD** layout and traversed as `(B,H,S,d)`.
  * `cost_sh`: number of cache lines accessed when the same data is stored in **BSHD** layout and traversed as `(B,S,H,d)`.

The implementation must use only Python for any arithmetic; no explicit Python loops over elements are allowed.  The result should be deterministic across platforms.

## Example

```python
shape = (2, 3, 4, 5)   # B=2, H=3, S=4, d=5
cost_bh, cost_sh = layout_cost(shape)
print(cost_bh, cost_sh)   # e.g. 120 140
```

## What the gate checks

The grader computes a reference implementation that follows the same stride‑based cache‑line counting logic.  
It then compares the tuple returned by your `layout_cost` to the reference using an exact integer match (`==`).  Any deviation causes the task to fail.
