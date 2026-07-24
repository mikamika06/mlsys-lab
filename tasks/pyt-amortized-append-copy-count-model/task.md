## Context

A Python list stores references in a contiguous array. Appending an item is usually
constant time, but occasionally the list must allocate a larger backing array and
copy the existing references.

For a sequence of $N$ appends, let $C(N)$ be the total number of element
references copied during reallocations. If a resize happens when the current
logical length is $k$, that resize contributes $k$ copies.

The over-allocation strategy means the list does not resize after every append.
Therefore the total copy work grows linearly:

$$
C(N) = \sum_{\text{resizes at length } k} k = O(N).
$$

The exact value depends on the CPython list implementation. This task models the
actual interpreter behavior rather than assuming a fixed growth formula.

## Task

Implement `append_copy_count(n)`:

```python
def append_copy_count(n: int) -> int:
    ...
```

Create an empty Python list and perform exactly $n$ `append` operations. Return
the total number of existing elements that would be copied because of backing
array reallocations.

The count should include a resize contribution of the old list length whenever
the list capacity changes after an append. Do not count the newly appended
element as a copy.

## Example

```python
print(append_copy_count(0))
# 0

print(append_copy_count(4))
# 0
```

Small lists may have no reallocations. The exact count for larger values is
determined by the running Python interpreter's list growth behavior.

## What the gate checks

The gate compares the returned value with an oracle built from the real CPython
list object. The oracle performs real appends and detects capacity changes using
`sys.getsizeof`, then sums the old lengths at each resize.

A hardcoded table of expected values does not pass because the reference values
are generated from the interpreter at grading time. The implementation must
produce the same copy-count model.
