## Context

In C‑like languages, a struct may contain padding bytes to satisfy alignment constraints. NumPy’s structured dtypes emulate this behaviour. When a dtype is created with `align=True` (the default), the interpreter inserts padding so that each field starts at an address that is a multiple of its natural alignment. The overall `itemsize` is thus typically larger than the sum of the individual field sizes.

For a packed representation we set `align=False`. This forces the fields to be laid out back‑to‑back, which reduces memory consumption but can incur unaligned accesses on some CPUs.

Consider a record with four fields:
- an unsigned 32‑bit integer ($4$ bytes),
- an unsigned 8‑bit integer ($1$ byte),
- a double‑precision floating point ($8$ bytes),
- and a boolean ($1$ byte).

If we lay them out without padding, the total size would be
$$4 + 1 + 8 + 1 = 14 \text{ bytes}.$$
With natural alignment on a 64‑bit machine, the fields are laid out as: `uint32` at offset $0$ (4 bytes), `uint8` at offset $4$ (1 byte), $3$ padding bytes so that `float64` starts at offset $8$ (its natural alignment), `float64` at offset $8$ (8 bytes), and `bool` at offset $16$ (1 byte) — giving $17$ bytes of used space. The struct's overall size must then be rounded up to a multiple of the strictest member alignment ($8$, from `float64`), so the final aligned layout becomes $24$ bytes.

## Task

Implement the function `packed_vs_aligned_ratio(n: int) -> float` that returns the ratio of the total memory footprint of a **naturally aligned** NumPy structured array of `n` records to that of a **packed** version with no padding. The ratio should be computed from the dtype’s `itemsize`, not by constructing the arrays.

```python
def packed_vs_aligned_ratio(n: int) -> float:
    ...
```

The function must work for any positive integer `n`. Use only NumPy (no explicit Python loops).

## Example

```python
>>> packed_vs_aligned_ratio(5)
1.7142857142857142  # aligned size / packed size = 24/14 ≈ 1.714
```

The returned value is a pure float; the input `n` does not affect the ratio.

## What the gate checks

The grader computes the exact reference ratio from NumPy’s dtype definitions and compares it to your output using an absolute difference metric. Your implementation must satisfy

$$|\text{output} - \text{reference}| \le 10^{-12}.$$

Any deviation larger than this threshold causes the task to fail.
