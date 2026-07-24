## Context

In a system with virtual memory each process sees a linear address space that is divided into pages of fixed size $P$ (commonly $4096$ bytes). A virtual address $a$ belongs to page number $\lfloor a / P \rfloor$. The set of distinct pages accessed during an execution trace is called the **page working set**.

## Task

Implement `count_distinct_pages(trace, page_size=4096)` that takes a 1‑D NumPy array `trace` containing byte‑addresses and returns the number of unique page numbers touched. Use vectorized NumPy only; no Python loops.

```python
def count_distinct_pages(trace: np.ndarray, page_size: int = 4096) -> int:
    ...
```

The result must be a plain `int`. The input may contain any unsigned integer type but you should cast it to an appropriate type.

## Example

```python
import numpy as np
trace = np.array([0, 4095, 8192, 12288, 16384, 20480], dtype=np.uint64)
print(count_distinct_pages(trace))          # 4
# page numbers: [0, 0, 2, 3, 4, 5] → distinct = 4
```

## What the gate checks

The grader supplies a set of deterministic test traces. Your implementation must return exactly the same integer as the reference implementation for every trace. No tolerance is applied; if any mismatch occurs the gate fails.
