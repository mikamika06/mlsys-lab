## Context

Production CUDA caching allocators never `cudaMalloc` the exact number of
bytes a tensor needs. Every request first gets **rounded up** to a fixed
block granularity, and is then routed to one of **two independent pools**
depending on how big the rounded request is — because small, frequent
allocations (activations, gradients) and large, infrequent ones (weight
buffers) fragment memory very differently if they share the same free
list.

Let $B = 512$ be the minimum block size. A raw request of $n$ bytes is
first rounded up:

$$
\mathrm{round}(n) = B \cdot \left\lceil \frac{\max(n, 1)}{B} \right\rceil .
$$

The rounded size then determines **which pool** services the request:

$$
\text{pool}(n) =
\begin{cases}
\text{small}, & \mathrm{round}(n) \le S \\
\text{large}, & \mathrm{round}(n) > S
\end{cases}
\qquad S = 1\,\mathrm{MiB} = 1048576 .
$$

Each pool carves memory out of **segments** — contiguous device regions it
grows on a cache miss. The segment size used depends on the rounded request
size too, with its own thresholds:

$$
\mathrm{segment}(n) =
\begin{cases}
2\,\mathrm{MiB}, & \mathrm{round}(n) \le S \\[4pt]
20\,\mathrm{MiB}, & S < \mathrm{round}(n) < L \\[4pt]
R \cdot \left\lceil \dfrac{\mathrm{round}(n)}{R} \right\rceil, & \mathrm{round}(n) \ge L
\end{cases}
$$

where $L = 10\,\mathrm{MiB} = 10485760$ (the point past which a request is
"large enough" to size its own segment) and $R = 2\,\mathrm{MiB} =
2097152$ (the rounding granularity used for those custom-sized segments).

Intuition: below 1 MiB, allocations are packed tightly into uniform 2 MiB
segments (cheap to reuse, low overhead per request). Between 1 MiB and
10 MiB, a request still gets a generously-sized 20 MiB segment so it has
room to be reused by other mid-sized requests later. At 10 MiB and above,
the request is big enough that giving it a segment sized to (a rounded-up
version of) itself wastes little space.

## Task

Implement:

```python
def route_allocation(nbytes: int):
    ...
```

Given a raw request size `nbytes` (a positive integer, in bytes), return
`(pool, segment_size)`:

* `pool` — the string `"small"` or `"large"`, chosen from the **rounded**
  request size as defined above.
* `segment_size` — the integer segment size (in bytes) that would be used
  to service this (rounded) request, as defined above.

## Example

```python
route_allocation(100)         # -> ("small", 2097152)     # rounds to 512, small pool, 2 MiB segment
route_allocation(1_000_000)   # -> ("small", 2097152)     # rounds to 1000448 <= 1 MiB
route_allocation(1_048_577)   # -> ("large", 20971520)    # rounds just over 1 MiB -> large pool, 20 MiB segment
route_allocation(15_000_000)  # -> ("large", 16777216)    # >= 10 MiB -> segment sized to itself, rounded up to a 2 MiB multiple
```

## What the gate checks

The grader evaluates `route_allocation` on every boundary value around each
threshold ($B$, $S$, $L$, and multiples of $R$) plus a large batch of
seeded random sizes spanning from a few bytes up to 200 MiB, and compares
against a NumPy/Python oracle implementing the exact rounding and
threshold logic above (never calling your function, never hardcoding an
expected table).

`exact_match` requires **every** case to return the exact `(pool,
segment_size)` pair the oracle computes — `1.0` if all match, `0.0` on the
first mismatch or exception. Using `<` instead of `<=` at a threshold,
rounding down instead of up, or swapping which pool a boundary value lands
in will all fail on the corresponding edge case even if most random inputs
still pass.
