## Context

On a GPU, threads run in **warps** of 32 lanes.  The memory subsystem has a fixed
**latency** $L$ cycles between issuing a load and receiving the data.  The warp
scheduler hides that latency by **switching to another ready warp** while the first
one waits.

By **Little's Law**, the minimum number of resident warps $W$ needed to keep the
pipeline fully busy is

$$W_{\min} = \left\lceil \frac{L}{I} \right\rceil$$

where $L$ is the round-trip latency in cycles and $I$ is the number of independent
instructions a single warp can issue before it stalls (i.e., the **instruction-level
parallelism** window per warp, often equal to the issue cadence between the load and
the next dependent instruction).

Intuitively: each warp covers $I$ cycles of useful work while in flight, so you need
at least $\lceil L / I \rceil$ warps to keep one instruction issuing every cycle.

## Task

Implement `min_warps_to_hide_latency(L: int, I: int) -> int` that returns the minimum
number of resident warps required to fully hide memory latency $L$ (in cycles) given
that each warp issues $I$ independent instructions before stalling.

$$W_{\min} = \left\lceil \frac{L}{I} \right\rceil$$

Both `L` and `I` are positive integers.

## Example

```python
# L = 200 cycles (typical L1/L2 miss), I = 10 independent instructions
min_warps_to_hide_latency(200, 10)  # -> 20
# L = 100, I = 7  -> ceil(100/7) = ceil(14.28...) = 15
min_warps_to_hide_latency(100, 7)   # -> 15
```

## What the gate checks

`check.py` generates several `(L, I)` pairs, computes
$\lceil L / I \rceil$ as the reference, and checks that your function returns the
**exact integer** for every case (`exact_match == 1.0`).
