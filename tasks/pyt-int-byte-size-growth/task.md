## Context

CPython stores arbitrary precision integers using a variable number of internal base-$2^{30}$ digits on typical 64-bit builds. As an integer grows, its object size increases when another internal digit is needed.

For a non-negative integer $n$, the number of base-$2^{30}$ digits is

$$
d(n) =
\begin{cases}
1, & n = 0, \\
\left\lceil \frac{\log_2(n+1)}{30} \right\rceil, & n > 0 .
\end{cases}
$$

The memory reported by `sys.getsizeof` includes the fixed object header plus storage for the internal digits. Small integers may be cached by CPython, but the reported object size is determined by the integer representation.

This task asks you to derive the size sequence by querying the runtime object model rather than by using a table of values.

## Task

Implement `int_size_growth()`:

```python
def int_size_growth() -> list[int]:
    ...
```

Return the list of `sys.getsizeof` values for these integers:

```python
[
    0,
    1,
    2**30 - 1,
    2**30,
    2**60 - 1,
    2**60,
    2**90,
]
```

The function must compute the sizes using the standard library, not by returning precomputed constants.

## Example

```python
sizes = int_size_growth()
# Example shape:
# [small_int_size, small_int_size, ..., larger_int_size]
```

The exact numbers depend on the pinned CPython build used by the evaluator.

## What the gate checks

The gate computes the expected sequence from the running CPython interpreter using `sys.getsizeof` on the specified integers. Your returned list must exactly match the oracle sequence.

Returning a hardcoded sequence will not be portable across CPython object layouts, while using the runtime object model will match the pinned interpreter.
