## Context

**Temporal locality** is the observation that a program which touches an
address once is likely to touch that same address again soon. The
simplest possible measurement of temporal locality in a trace doesn't
need a cache model at all, and doesn't need capacity, associativity, or
an eviction policy — it only needs one question per access: *has this
exact address ever appeared earlier in the trace?* An access that
answers "yes" is a **reuse**; an access to a brand-new address is a
**first touch**, and first touches are never reuses, no matter how many
times that address shows up later.

## Task

Implement

```cpp
long long count_reuses(const long* addrs, int n);
```

`addrs[0..n)` is a trace of byte addresses. For each index `i`, the
access is a reuse if `addrs[i]` equals `addrs[j]` for at least one
`j < i`. Return the total number of reuse accesses in the trace (the
first occurrence of every distinct address is excluded from the count).

## Example

Trace `[0, 8, 16, 24, 32, 40, 48, 56, 0, 8, 16, 24]`:

- The first 8 accesses (`0, 8, 16, 24, 32, 40, 48, 56`) are all first
  touches — 0 reuses so far.
- `0`, `8`, `16`, `24` then repeat — each of those 4 accesses is a reuse.

`count_reuses` returns `4` for this trace.

## What the gate checks

`main.cpp` generates a fixed 40-access trace over a working set of 10
distinct byte addresses using a seeded generator (reproducible on every
run), calls `count_reuses`, and prints the result. The grader compiles
your `.cpp` with the real local `clang++`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{your printed count matches the reference's exactly}
$$

Returning `0` (or any other constant) fails immediately, since a 40-long
trace over only 10 distinct addresses is guaranteed to contain many
reuses — you actually have to track which addresses have already been
seen.
