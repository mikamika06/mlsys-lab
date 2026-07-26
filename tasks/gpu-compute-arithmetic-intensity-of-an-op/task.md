## Context

The arithmetic intensity (AI) of a GPU kernel is defined as the number of floating‑point operations performed per byte moved from global memory:

$$
\text{AI} = \frac{\text{FLOPs}}{\text{bytes}_{\text{read}} + \text{bytes}_{\text{written}}},
$$

where $\text{bytes}_{\text{read}}$ and $\text{bytes}_{\text{written}}$ are the total sizes of data transferred to and from global memory during a single kernel launch. A high AI indicates compute‑bound behaviour, whereas a low AI indicates that the kernel is memory‑bandwidth limited.

## Task

Implement `arithmetic_intensity` that receives:

- `flops`: an integer number of floating‑point operations performed by the kernel,
- `read_bytes`: an integer number of bytes read from global memory,
- `write_bytes`: an optional integer number of bytes written to global memory (default 0),

and returns the arithmetic intensity as a `float`. The returned value must be rounded to **six decimal places**. If no bytes are transferred, return `float('inf')`.

```python
def arithmetic_intensity(flops: int,
                         read_bytes: int,
                         write_bytes: int = 0) -> float:
    ...
```

## Example

```python
>>> arithmetic_intensity(500, 100, 200)
1.666667

>>> arithmetic_intensity(123456, 1111, 2222)
37.040504

>>> arithmetic_intensity(10, 5, 5)
1.000000

>>> arithmetic_intensity(0, 0, 0)
inf
```

## What the gate checks

The grader compares your function’s output to a reference implementation on several test cases using exact equality after rounding to six decimal places. The metric is `exact_match` with threshold = 1.0. A single mismatch fails the task. No performance or runtime restrictions are enforced beyond correctness.
