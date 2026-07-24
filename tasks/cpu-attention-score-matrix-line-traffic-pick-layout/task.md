## Context

Computing an attention score matrix $S = QK^\top$ (query matrix $Q \in
\mathbb{R}^{n \times d}$, key matrix $K \in \mathbb{R}^{n \times d}$) reads
every row of $Q$ against every row of $K$, $n^2$ dot products in total.
How $K$ is laid out in memory changes how many DRAM cache lines that reads,
even though the *bytes touched* (the underlying data) are identical:

- **Row-major** $K$ (shape $[n, d]$, same as $Q$): key vector $j$'s $d$
  components are **contiguous** — they pack into a handful of cache lines
  that get reused efficiently for every $i$.
- **Transposed** $K^\top$ (shape $[d, n]$): key vector $j$'s $d$ components
  are **strided** by $n \cdot 4$ bytes apart — each component can land in
  its own cache line, and because $Q$ and $K$ together don't fit in a small
  cache, this scatter causes far more line evictions and re-fetches as $i$
  advances.

Real hardware cache misses aren't reproducible across machines, so this
task grades against a **deterministic model**: a fixed direct-mapped
cache (16KB, 64-byte lines) that every memory touch is routed through, via
`touch_byte(addr)` (declared in `sol.hpp`, defined in `main.cpp`).

## Task

Implement two functions in `solve.cpp`:

```cpp
void simulate_score_matrix_traffic(int layout, int seq_len, int head_dim,
                                    long q_base, long k_base);
int pick_better_layout(int seq_len, int head_dim);
```

`simulate_score_matrix_traffic`: for every $i, j \in [0, \text{seq\_len})$
and every $d \in [0, \text{head\_dim})$, call `touch_byte` for the byte
address of $Q[i][d]$ (always row-major: `q_base + (i*head_dim + d)*4`) and
for the byte address of $K[j][d]$ under the requested layout:

- `layout == 0`: `k_base + (j*head_dim + d)*4` (row-major, contiguous)
- `layout == 1`: `k_base + (d*seq_len + j)*4` (transposed, strided)

`pick_better_layout`: call `reset_cache()`, run
`simulate_score_matrix_traffic` for `layout=0`, read `miss_count()`; repeat
for `layout=1`; return `0` if row-major produced fewer-or-equal misses,
`1` if the transposed layout did.

## Example

At `seq_len=64, head_dim=64` (a $Q$/$K$ pair that together exceed the 16KB
model cache), the reference measures `1264` misses for row-major $K$ versus
`8823` misses for transposed $K$ — about $7\times$ more DRAM traffic from
the transpose alone, despite touching the exact same underlying data. At
small sizes that fit comfortably in cache (e.g. `seq_len=16, head_dim=8`),
the two layouts tie — the effect is a capacity/eviction phenomenon, not a
constant per-byte penalty.

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across three `(seq_len,
head_dim)` fixtures — two where the layouts tie and one where the
transposed layout is dramatically worse. The starter never calls
`touch_byte` at all, so every printed miss count stays `0` and the layout
choice is meaningless.
