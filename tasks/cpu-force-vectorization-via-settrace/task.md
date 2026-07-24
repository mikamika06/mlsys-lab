## Context

Modern processors use vector units (SIMD) to perform multiple operations per instruction.
Accessing memory with unit stride (consecutive elements) allows the hardware to stream data
into vector registers efficiently. On the software side, Python’s bytecode interpreter
executes every source line as an event. Using explicit `for` loops over array elements
generates many line events, wasting CPU time. Vectorized operations through libraries like
NumPy avoid per-element loops and drastically reduce line events.

In this task you generate the order in which to visit every element of a row-major
$n \times n$ `float64` matrix. A sequential row-by-row order is both vector-friendly
(unit stride) and cache-friendly (one miss per 64‑byte cache line). The grader counts
Python line events during your function call; a low count indicates a vectorized
implementation. It also replays your order through a cache simulator to confirm the
access pattern is cache-efficient.

## Task

Implement

```python
def access_pattern(n: int) -> list[int]:
    """Return a permutation of 0..n^2-1 that visits every matrix element once,
       row by row, without Python loops."""
```

Your implementation **must not** contain explicit `for`/`while` loops or list comprehensions
that iterate over the elements; use NumPy’s `arange` or plain `range` outside a loop.
The grader uses `sys.settrace` to count Python line events; if the count exceeds 10 your
solution is rejected.

## Example

```python
access_pattern(2)  # -> [0, 1, 2, 3]
```

## What the gate checks

The cache simulator runs with a 64‑byte line, 64 sets, 8 ways. With $n=64$ a sequential
order produces $\frac{64^2}{8}=512$ compulsory misses. The gates are:

- `covers_all` = 1.0 iff the list contains every index $0\ldots n^2-1$ once.
- `line_count` ≤ 10  (Python line events executed inside `access_pattern`).
- `misses` ≤ 512  (cache simulator result).

All three must pass.
