## Context

Modern CPUs contain a **hardware prefetcher** that monitors memory access
patterns and speculatively loads data before it is needed. A stream+stride
prefetcher tracks the byte-address stride between consecutive accesses: once
the same stride repeats, it starts issuing loads one stride ahead and keeps
doing so as long as the stride stays constant. It gives up the moment the
stride changes, and real prefetchers also stop tracking once a stride would
carry them across a page boundary (page size = 4096 B here).

A pattern is therefore **caught** iff its trace has a single, constant,
nonzero stride whose magnitude is strictly below one page:

- **Sequential** access (stride = element size): caught.
- **Fixed stride** (e.g. every 4th element): caught, as long as the stride is
  small.
- **Random** (shuffled indices): the stride jumps around every step — not
  caught.
- **Pointer chasing** (linked list, next address depends on the value just
  loaded): the addresses have no fixed stride either — not caught.
- **Stride $\ge$ page size** (4096 B): even though the stride is constant,
  hardware prefetchers stop tracking across page boundaries — not caught.

## Task

Implement `classify_prefetch` (declared in `sol.hpp`):

```cpp
void classify_prefetch(const long* const* addrs, const int* lens, int num_patterns, int* out);
```

`addrs[k]` points to `lens[k]` byte addresses for pattern $k$. For each
pattern, decide whether a stream+stride prefetcher would catch it and write
`1` (caught) or `0` (not caught) into `out[k]`:

$$\text{caught}_k \;=\; \Bigl[\forall i:\; a_k[i+1]-a_k[i] = s_k\Bigr] \;\wedge\; s_k \neq 0 \;\wedge\; |s_k| < 4096$$

where $s_k = a_k[1] - a_k[0]$ is the trace's first stride.

`main.cpp` builds 5 fixed traces (4-byte elements): sequential (stride 4 B,
256 elements), fixed stride (stride 16 B, 64 elements), a deterministic
"random" permutation (irregular stride, 256 elements), a deterministic
"pointer chase" permutation (a different irregular stride, 256 elements),
and a large stride (4096 B, 64 elements).

## Example

```
classify_prefetch(...)
// out[0] = 1   (sequential — constant 4 B stride)
// out[1] = 1   (fixed stride — constant 16 B stride)
// out[2] = 0   (random — stride varies every step)
// out[3] = 0   (pointer-chase — stride varies every step)
// out[4] = 0   (constant stride, but 4096 B == one page)
```

## What the gate checks

The gate is `exact_match` on the driver's printed output. It compiles your
`classify_prefetch` against the fixed 5-pattern driver, runs it, and compares
stdout to the reference's stdout line for line. Marking every constant-stride
trace as caught regardless of magnitude (missing the page-size cutoff on
pattern 4), or checking only the FIRST stride instead of every consecutive
one (wrongly catching the random/pointer-chase patterns), both flip at least
one label and fail the gate.
