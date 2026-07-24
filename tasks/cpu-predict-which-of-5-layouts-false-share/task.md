## Context

Modern CPUs cache memory in 64-byte **cache lines**. When two threads write to
*different* variables that happen to occupy the **same** cache line, the hardware
coherence protocol forces the line to bounce between cores — a phenomenon called
**false sharing**. No actual data is shared, yet performance degrades as if it
were.

A layout causes false sharing when objects owned by *different threads* alias
onto the same 64-byte line. Formally, thread $t_i$ owns object at base address
$a_i$. If

$$\lfloor a_i / 64 \rfloor = \lfloor a_j / 64 \rfloor \quad (i \neq j)$$

then threads $t_i$ and $t_j$ false-share.

The five candidate layouts store per-thread counters (8-byte `int64`) at
different strides. Your job is to label each layout `True` (false-sharing occurs)
or `False` (each thread's counter lives on its own cache line).

## Task

Implement `classify_layouts(line_bytes: int) -> list[bool]`, which returns a
list of five booleans. Element $k$ is `True` if layout $k$ causes false sharing
(two or more of the 4 threads touch the same cache line), `False` otherwise.

The five layouts are fixed — they describe where thread $t \in \{0,1,2,3\}$
places its 8-byte counter:

| Layout | Byte address of thread $t$'s counter |
|--------|--------------------------------------|
| 0 | $t \times 8$ (packed, stride = 8 B) |
| 1 | $t \times 64$ (stride = 1 line) |
| 2 | $t \times 128$ (stride = 2 lines) |
| 3 | $t \times 8 + 64 \times (t \% 2)$ (alternating line offset) |
| 4 | $t \times 16$ (stride = 16 B) |

## Example

With `line_bytes = 64`, layout 0 places all four counters at bytes 0, 8, 16,
24 — all within the same 64-byte line, so it **false-shares** → `True`.
Layout 1 places counters at 0, 64, 128, 192 — one per line → `False`.

```python
result = classify_layouts(64)
# result[0] == True   (all 4 in one line)
# result[1] == False  (one per line)
```

## What the gate checks

`check.py` recomputes the reference labels using the line-overlap formula
$\lfloor a_i / \text{line\_bytes} \rfloor$ for all thread pairs in each layout,
then checks `exact_match` — your returned list must agree with the reference
on all five layouts.
