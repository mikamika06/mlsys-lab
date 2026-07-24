## Context

A Python list stores references in a dynamically managed array. When elements are
added beyond the current capacity, CPython may allocate a larger backing array
and copy references into the new storage. Each such growth event is a capacity
change.

The memory footprint reported by `sys.getsizeof` includes the list object and
its currently allocated pointer storage. For a fixed interpreter build, a change
in this value during construction indicates that the list storage changed.

If a list grows one element at a time, the number of reallocations can be much
larger than if the final size is known in advance. The goal is to construct the
same logical list while staying within a very small reallocation budget.

## Task

Implement `build_list_realloc_count(n)`.

The function receives a non-negative integer $n$. It must build a list containing
the values $0, 1, \dots, n-1$ and return the number of list storage size changes
observed while constructing it.

The count is measured by comparing successive `sys.getsizeof` observations. A
change is counted whenever two consecutive observations differ.

The implementation should avoid repeated list growth. It is allowed to use
normal Python list operations, but the returned value must represent the number
of reallocations for the construction strategy used.

The function signature is:

```python
def build_list_realloc_count(n: int) -> int:
    ...
```

## Example

```python
count = build_list_realloc_count(1000)
# A preallocated construction can keep this count at zero.
```

## What the gate checks

The gate computes the expected answer using a CPython oracle that constructs the
same sized list with a capacity-known strategy and measures size changes using
`sys.getsizeof`.

Your function is called on several values of $n$. It must return the same count
as the oracle for every case.

A strategy that repeatedly appends elements will usually observe multiple size
changes because the list capacity grows in steps.
