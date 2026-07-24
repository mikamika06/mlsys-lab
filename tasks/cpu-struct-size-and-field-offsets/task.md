## Context

In C/C++ a **struct** is laid out in memory by placing each field at the smallest address that satisfies its alignment requirement, and padding may be added between fields or after the last field so that the size of the struct itself is a multiple of the strictest alignment among its members.  
The natural alignment of a type is typically equal to its size (e.g., `int` has size 4 bytes and must start at an address divisible by 4). This padding is invisible in high‑level code but matters for binary compatibility, cache performance, and interfacing with other languages.

For example, consider the following struct:

```c
struct S {
    char   a;     // offset 0, size 1
    int    b;     // needs alignment 4 → offset 4
    short  c;     // needs alignment 2 → offset 8
};
```

After `b` we are at byte 8, which is already aligned for a `short`. The total size of the struct must be a multiple of its maximum alignment (here 4), so we add two padding bytes after `c`, giving a final size of **12** bytes.

The task below asks you to implement this layout computation in Python.

## Task

Implement the function

```python
def compute_struct_layout(field_types: List[str]) -> Tuple[List[int], int]:
    ...
```

`field_types` is a list of strings representing the C type names (one of `"char"`, `"short"`, `"int"`, `"long"`, `"float"`, or `"double"`).  
The function must return a tuple:

1. A list of offsets (in bytes) for each field, in the order given.
2. The total size of the struct, including any padding required to satisfy alignment rules.

You should use only Python's standard library; no external packages are needed.  The implementation must be deterministic and run quickly on small inputs.

## Example

```python
>>> compute_struct_layout(["char", "int", "short"])
([0, 4, 8], 12)
```

The first field `char` starts at offset 0.  
`int` is aligned to 4 bytes → offset 4.  
After that we are at byte 8, which satisfies the alignment of `short` (2), so it gets offset 8.  
The struct’s maximum alignment is 4, and after placing a 2‑byte `short` the current size is 10, requiring two padding bytes to reach the next multiple of 4 → total size 12.

## What the gate checks

A single gate named **exact_match** verifies that the tuple returned by your function matches the reference implementation exactly for several test cases.  No other metrics are used; the solver must simply produce correct offsets and size for each supplied list of field types.
