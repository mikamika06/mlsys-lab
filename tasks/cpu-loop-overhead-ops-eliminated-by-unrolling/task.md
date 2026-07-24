## Context

A counted `for` loop performs **overhead operations** on every iteration that are
not part of the useful body. On a typical scalar pipeline these are:

- **Counter increment**: `i += 1` (one ALU operation)
- **Conditional branch**: `i < N ?` (one compare-and-branch)

That gives $K = 2$ overhead operations per iteration. For a loop that executes
$N$ iterations the total overhead is $K \cdot N = 2N$ operations.

**Loop unrolling** by a factor $U$ (where $U$ divides $N$) places $U$ copies of
the body inside each iteration, shrinking the iteration count from $N$ to
$N/U$. The overhead drops to $2 \cdot N/U$, and the number of overhead
operations *eliminated* is

$$\Delta = 2 \!\left(N - \frac{N}{U}\right)$$

The data-element accesses of the body remain exactly $N$; only the
bookkeeping shrinks.

## Task

Implement `unroll_overhead(N: int, U: int) -> tuple[int, list[int]]`.

- `N` — original iteration count ($U$ divides $N$, both positive).
- `U` — unroll factor ($U \ge 1$).

Return `(saved_ops, trace)`:

1. **`saved_ops`** (`int`): the exact number of overhead operations eliminated,
   using the $K = 2$ model above.
2. **`trace`** (`list[int]`): exactly `N` byte addresses representing the
   unrolled loop body's data accesses. Element $i$ (for
   $i \in 0 \ldots N{-}1$) should reside at byte address $8\,i$ (sequential
   `float64` layout). The access order inside each group of $U$ consecutive
   elements is flexible, but every address must appear exactly once.

## Example

```python
unroll_overhead(8, 2)
# saved_ops = 2 * (8 - 4) = 8
# trace = [0, 8, 16, 24, 32, 40, 48, 56]   # one valid ordering
```

## What the gate checks

| Gate | Condition |
|------|-----------|
| `exact_match` | `saved_ops` equals $2(N - N/U)$ **exactly** ($N{=}1024$, $U{=}4$, so the answer is $1536$) |
| `covers_all` | `trace` is a permutation of $\{0, 8, 16, \ldots, 8184\}$ |
| `misses` | Cache simulator replays `trace` through a 64-set, 8-way, 64-B-line cache; miss count $\le 200$ (sequential access triggers only compulsory misses) |
