## Context

CPython's pymalloc is a specialised allocator for small Python objects. Memory is
organised in a three-level hierarchy:

1. **Arena** – a contiguous 256 KiB ($2^{18}$ bytes) region obtained from the OS.
2. **Pool** – a 4 KiB ($2^{12}$ bytes) page within an arena. Each pool is assigned
   to exactly one *size class* and serves blocks of that size.
3. **Block** – the unit of allocation handed to `PyObject_Malloc`. All blocks in a
   given pool share the same size.

A pool begins with a fixed-size header that stores bookkeeping fields (free-list
pointer, size-class index, arena index, etc.). The remaining bytes are carved into
blocks:

$$\text{blocks\_per\_pool}(s) \;=\; \left\lfloor \frac{P - H}{s} \right\rfloor$$

where $P$ is the pool size (typically $4096$ bytes), $H$ is the pool-header size,
and $s$ is the block size for the chosen size class.

Size classes are positive multiples of an alignment constant $A$ (typically $8$
bytes on 64-bit systems), ranging from $A$ up to a maximum block size $B$:

$$\mathcal{S} \;=\; \bigl\{\,A,\; 2A,\; 3A,\; \dots,\;
\bigl\lfloor B / A \bigr\rfloor \cdot A\,\bigr\}$$

For example, with $A = 8$, $P = 4096$, $H = 16$, and $B = 512$:

- Size class $s = 8$: $\lfloor(4096 - 16)/8\rfloor = 509$ blocks
- Size class $s = 24$: $\lfloor(4096 - 16)/24\rfloor = 169$ blocks
- Size class $s = 512$: $\lfloor(4096 - 16)/512\rfloor = 7$ blocks

There are $B / A = 64$ size classes in total, and each produces a different
blocks-per-pool count.

## Task

Implement `blocks_per_pool`:

```python
def blocks_per_pool(alignment: int, pool_size: int, pool_header_size: int,
                    max_block_size: int) -> list[int]:
    ...
```

The function receives the allocator's constants and must return a list of
blocks-per-pool values — one per size class, in ascending order of block size.

The size classes are the positive multiples of `alignment` that do not exceed
`max_block_size`: $s \in \{a,\; 2a,\; 3a,\; \dots\}$ where $a$ is `alignment`
and the largest class satisfies $s \leq \texttt{max\_block\_size}$.

For each such $s$, compute

$$\left\lfloor \frac{\texttt{pool\_size} - \texttt{pool\_header\_size}}{s}
\right\rfloor$$

using integer (floor) division.

## Example

```python
# 64-bit CPython defaults
blocks_per_pool(alignment=8, pool_size=4096, pool_header_size=16, max_block_size=512)
# => [509, 254, 169, 127, 101, ..., 14, 12, 11, 10, 9, 8, 7]
#     s=8  s=16 s=24 s=32 s=40      ...                     s=512
# Length = 512 // 8 = 64
```

```python
# Edge: max_block_size equals alignment → single size class
blocks_per_pool(alignment=256, pool_size=4096, pool_header_size=16, max_block_size=256)
# => [15]
```

## What the gate checks

One gate: `exact_match`. The grader independently derives the reference answer
by iterating over size classes as multiples of `alignment` up to
`max_block_size` and computing $\lfloor(P - H)/s\rfloor$ for each. It then
compares the full list element-by-element. Any off-by-one in the integer
division, any wrong stop condition, or any missing/extra size class flips the
gate to 0.
