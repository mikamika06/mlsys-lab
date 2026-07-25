## Context

A **working-set curve** plots, for every window size $w$, how many
*distinct* cache lines appear in the most recent $w$ memory accesses of a
trace. Grow $w$ from $1$ and the curve climbs — every new address added
to the window has a decent chance of being one you haven't seen yet —
until $w$ reaches the trace's true **working-set size** $K$ (the number
of distinct lines it ever touches). Past that point every additional
address the window picks up is necessarily a *repeat* of a line already
counted, so the curve goes flat: a **plateau**. Where the plateau starts
tells you $K$ without ever being told it directly — exactly the
measurement a real "sweep working-set size, watch when performance stops
changing" cache-sizing experiment is doing.

## Task

Implement

```cpp
int plateau_index(const long* addrs, int n, int max_w, int line_bytes, int* curve_out);
```

For every window size $w$ in $[1, \text{max\_w}]$, look at the **last**
$w$ elements of the trace, $\mathrm{addrs}[n-w \,..\, n-1]$, count how
many distinct cache lines ($\lfloor \mathrm{addr} / \mathrm{line\_bytes}
\rfloor$) appear among them, and write that count into
`curve_out[w-1]`. The curve is non-decreasing in $w$. Return the smallest
$w$ (1-indexed) such that `curve_out[w-1]` already equals the curve's
final value `curve_out[max_w-1]`.

## Example

A trace that cycles through addresses at line indices `0,1,2,0,1,2,0,1,2`
(K=3, repeated 3 times, `line_bytes` given): the last-`1` window sees 1
distinct line; the last-`2` window sees 2; the last-`3` window (and every
larger one) sees all 3 and never grows past that — `plateau_index = 3`.

## What the gate checks

`main.cpp` builds two fixed traces, each with a WARMUP prefix of
distinct, never-repeated line addresses followed by a STEADY region that
cycles through `K` distinct 64-byte-line addresses whose byte-level
offset *within* each line also varies step to step (so distinct raw
addresses outnumber distinct lines) — `K=10` (`max_w=150`, entirely
inside the steady region) and `K=6` (`max_w=60`, likewise) — and prints
the plateau index plus a few curve samples for each. The candidate's full
stdout is compared byte-for-byte (`exact_match = 1.0`) against the
reference's. On these fixtures the reference recovers `plateau=10` and
`plateau=6` — exactly the true `K` of each trace, purely from watching
where the distinct-line count stops changing. Counting distinct *raw
addresses* instead of distinct *lines* (skipping the divide by
`line_bytes`) reports `plateau=30` instead of `10` on the first fixture,
since the varying offset multiplies how many distinct values there are
to see; scanning the window from the FRONT of the trace instead of the
last `w` elements picks up the (deliberately disjoint) warmup addresses
too, reporting `plateau=25` and `plateau=14` instead of `10` and `6`.
