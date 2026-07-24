## Context

A **software prefetch** is an explicit instruction (`__builtin_prefetch`
in GCC/Clang, `_mm_prefetch` on x86) that asks the memory system to start
fetching an address *before* the program actually needs it, so the
latency is paid while the CPU is busy doing other work instead of
stalling.

In a streaming loop that touches a large array with no reuse, a common
technique is to prefetch `distance` iterations ahead: while iteration
$i$ runs its own work, it also issues a prefetch for the element
iteration $i + \mathrm{distance}$ will need.

For the prefetch to actually hide the latency, it needs enough elapsed
time between being issued and being needed. If each loop iteration takes
$C$ cycles of work and a memory access takes $L$ cycles to complete, the
prefetch issued at iteration $i$ for iteration $i+\mathrm{distance}$ has
$\mathrm{distance} \cdot C$ cycles to finish before it's needed. The
minimum sufficient distance is therefore

$$
\mathrm{distance}_{\min} = \left\lceil \frac{L}{C} \right\rceil
$$

Too small a distance wastes the effort entirely (the data still isn't
ready, so the access stalls exactly as if there were no prefetch at
all). Too large a distance is not free either: the very first
`distance` iterations of the loop have no earlier iteration to have
issued their prefetch from, so they always stall on a cold access
regardless of how the memory system is doing -- a bigger distance simply
means a longer warm-up tax before the technique starts paying off.

## Task

Implement:

```cpp
int count_stalls(int n, int distance, int latency_cycles, int cycles_per_iter);
```

Return the number of stalling iterations (out of `n`) for a streaming
loop of `n` cold accesses, prefetching `distance` iterations ahead,
where each iteration costs `cycles_per_iter` cycles and a memory access
costs `latency_cycles` cycles:

- `warmup = min(distance, n)` iterations always stall (no earlier
  iteration existed to prefetch them).
- If `distance * cycles_per_iter >= latency_cycles`, every iteration
  from `distance` onward is fully hidden: return `warmup`.
- Otherwise the prefetch never finishes in time for *any* iteration:
  return `n` (every access still stalls, exactly as with no prefetching
  at all).

## Example

With `latency_cycles = 180`, `cycles_per_iter = 20`, and `n = 64`:
`distance_min = ceil(180 / 20) = 9`.

- `distance = 8`: `8 * 20 = 160 < 180` -- insufficient. `count_stalls`
  returns `64` (no benefit whatsoever, identical to not prefetching).
- `distance = 9`: `9 * 20 = 180 >= 180` -- sufficient. `count_stalls`
  returns `min(9, 64) = 9` (only the warm-up iterations stall).
- `distance = 40`: also sufficient (`800 >= 180`), but
  `count_stalls` returns `min(40, 64) = 40` -- more than four times the
  stalls of the optimal `distance = 9`, purely from an oversized warm-up
  window.

## What the gate checks

`main.cpp` fixes `n = 64`, `latency_cycles = 180`, `cycles_per_iter =
20` and calls `count_stalls` at distances `{1, 5, 8, 9, 12, 40}` --
three below `distance_min = 9`, three at or above it -- printing the
stall count for each. The candidate's full stdout is compared
byte-for-byte (`exact_match = 1.0`) against the reference's. A solution
that treats any positive `distance` as if it always fully hides the
latency (ignoring the `distance * cycles_per_iter >= latency_cycles`
condition) matches on the three sufficient distances but is wrong on all
three insufficient ones (it would report `1`, `5`, `8` stalls instead of
the true `64`, `64`, `64` -- prefetching too close to the deadline buys
nothing at all).
