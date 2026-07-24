## Context

A Python list stores references to objects in an internal array. The logical length of
a list, $n$, is the number of elements currently visible to Python, while the allocated
capacity is the number of reference slots reserved internally.

When a list grows with `append`, CPython usually allocates more space than immediately
needed. This reduces the number of reallocations. The memory reported by
`sys.getsizeof` includes a fixed object header plus the bytes for the internal pointer
array.

For a list with capacity $c$ on this CPython build,

$$
\mathrm{getsizeof}(L) = H + 8c,
$$

where $H$ is the list object header size and each stored reference occupies 8 bytes.
The header can be recovered from an empty list because its capacity is zero.

## Task

Implement `reconstruct_capacity(n)`:

```python
def reconstruct_capacity(n: int) -> list[int]:
    ...
```

Create an initially empty list and append integers until it contains $n$ elements.
After every append, use `sys.getsizeof` to infer the currently allocated slot count.

Return a list containing the inferred capacity after each append. The returned list must
have length $n$. Do not use implementation-specific constants for the header size.

## Example

```python
reconstruct_capacity(5)
# Example output on the target CPython build:
# [4, 4, 4, 4, 8]
```

The exact growth pattern depends on the CPython version and build. The grader computes
the expected sequence from the running interpreter.

## What the gate checks

The gate creates several list growth sequences and computes the expected capacities
using the real CPython `sys.getsizeof` behavior. Your function must return exactly the
same slot sequence.

A solution that returns only the logical length, or that hardcodes a growth table from
another Python version, will fail.
