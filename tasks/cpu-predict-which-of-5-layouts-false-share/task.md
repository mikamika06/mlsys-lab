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
different strides. Your job is to label each layout `true` (false-sharing
occurs) or `false` (each thread's counter lives on its own cache line).

## Task

Implement

```cpp
std::array<bool, 5> classify_layouts(long line_bytes);
```

which returns five booleans. Element $k$ is `true` if layout $k$ causes false
sharing (two or more of the 4 threads touch the same cache line), `false`
otherwise.

The five layouts are fixed — they describe where thread $t \in \{0,1,2,3\}$
places its 8-byte counter:

| Layout | Byte address of thread $t$'s counter |
|--------|--------------------------------------|
| 0 | $t \times 8$ (packed, stride = 8 B) |
| 1 | $t \times 64$ (stride = 1 line) |
| 2 | $t \times 128$ (stride = 2 lines) |
| 3 | $t \times 8 + 64 \times (t \bmod 2)$ (alternating line offset) |
| 4 | $t \times 16$ (stride = 16 B) |

## Example

With `line_bytes = 64`, layout 0 places all four counters at bytes 0, 8, 16,
24 — all within the same 64-byte line, so it **false-shares** → `true`.
Layout 1 places counters at 0, 64, 128, 192 — one per line → `false`.

```
classify_layouts(64)
// result[0] == true   (all 4 in one line)
// result[1] == false  (one per line)
```

## What the gate checks

`main.cpp` calls `classify_layouts(64)` and prints one `0`/`1` per line, in
layout order. The grader compiles your `.cpp` with the real local `clang++`,
runs it, and compares against `ref.cpp`, which recomputes the reference
labels from the line-overlap formula $\lfloor a_i / \text{line\_bytes}
\rfloor$ over every thread pair in each layout:

$$
\mathrm{exact\_match} = 1 \iff \text{all five printed labels match the reference}
$$

The correct labeling is `[true, false, false, true, true]` — note layout 3
(alternating offset) and layout 4 (16-byte stride, four counters packed into
one 64-byte line) both false-share despite looking "spread out" at a glance;
a starter that just returns all-`false` fails on 3 of the 5 layouts.
