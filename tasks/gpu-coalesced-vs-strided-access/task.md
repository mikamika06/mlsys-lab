## Context

In NVIDIA GPUs, a warp consists of 32 threads that issue memory requests in parallel. Global memory is fetched in fixed-size transactions of 128 bytes (32 × 4 bytes for float32). If the addresses requested by the 32 threads fall within the same 128‑byte segment, the hardware issues a single transaction; otherwise each distinct segment incurs an additional transaction.

For a one‑dimensional array of `float32` elements, the address of element `i` is

$$
\text{addr}(i) = \text{base} + i \times 4,
$$

where `base` is the starting byte offset of the array. A warp that starts at index `0` and accesses indices spaced by a stride `s` will touch elements

$$
i_s(t) = t \cdot s, \qquad t=0,\dots,31.
$$

The number of 128‑byte segments touched is therefore

$$
\left|\Bigl\{\,\bigl\lfloor i_s(t)/32\bigr\rfloor : t=0,\dots,31\,\Bigr\}\right|.
$$

This quantity equals the number of memory transactions required for that warp.

## Task

Implement `count_transactions(arr: list[float], stride: int) -> int`:

```python
def count_transactions(arr, stride):
    ...
```

The function receives a list of floats of type `float32`. It should return the number of 128‑byte memory transactions that a single warp would perform when each thread reads an element spaced by `stride` starting at index 0. The result must be an integer.

## Example

```python
arr = list(range(128))

# stride 1 → indices 0–31; all in the first segment
print(count_transactions(arr, 1))   # 1

# stride 2 → indices 0,2,…,62; two segments
print(count_transactions(arr, 2))   # 2

# stride 3 → indices 0,3,…,93; three segments
print(count_transactions(arr, 3))   # 3
```

## What the gate checks

The grader computes a reference transaction count using the exact formula above and compares it to your output. The metric `exact_match` must equal 1.0 for all test cases.
