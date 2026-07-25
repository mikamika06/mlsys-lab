## Context

**Memory-level parallelism (MLP)** is how many memory requests a core
keeps outstanding at once. Pointer chasing (walking a linked list, tree,
or hash-table bucket) is the classic *adversarial* case: each hop
depends on the value the previous load returned, so without help the
requests are fully serialized and MLP collapses to 1 no matter how deep
the memory hierarchy's queues go.

Software prefetching restores parallelism by looking `distance` hops
ahead: while working on hop $i$, the loop also issues a prefetch for hop
$i + \mathrm{distance}$. By **Little's Law**, the number of requests
that must be in flight simultaneously to fully hide a latency $L$ at an
issue rate of one request per $C$ cycles is

$$
\text{required\_concurrency} = \left\lceil \frac{L}{C} \right\rceil
$$

But real hardware can only track a *fixed* number of outstanding memory
requests -- its miss-status-holding registers (MSHRs), `mlp_max`. Once
`required_concurrency` exceeds `mlp_max`, prefetching further ahead
cannot help any more: the requests you'd add just queue up behind the
ones already in flight. The loop becomes **MLP-bound** instead of
latency-bound, and some exposed latency is unavoidable no matter the
distance chosen.

## Task

Implement:

```cpp
int min_saturating_distance(int latency_cycles, int cycles_per_iter, int mlp_max);
bool is_latency_fully_hidden(int latency_cycles, int cycles_per_iter, int mlp_max);
```

1. `required_concurrency = ceil(latency_cycles / cycles_per_iter)`.
2. `min_saturating_distance` returns
   `min(required_concurrency, mlp_max)` -- there is never a reason to
   prefetch further ahead than whichever of the two limits binds first.
3. `is_latency_fully_hidden` returns `required_concurrency <= mlp_max`
   -- whether that minimum distance actually achieves full hiding
   (latency-bound regime) or is instead capped short of it by the
   hardware (MLP-bound regime).

## Example

`latency_cycles = 400`, `cycles_per_iter = 25`, `mlp_max = 10`:
`required_concurrency = ceil(400/25) = 16`. Since `16 > 10`, the
hardware's 10 outstanding requests are the binding constraint:
`min_saturating_distance` returns `min(16, 10) = 10`, and
`is_latency_fully_hidden` returns `false` -- even at maximum useful
distance, this loop is MLP-bound and still exposes latency every
iteration.

Contrast `latency_cycles = 100`, `cycles_per_iter = 50`,
`mlp_max = 10`: `required_concurrency = ceil(100/50) = 2`, and
`2 <= 10`, so `min_saturating_distance` returns `2` and
`is_latency_fully_hidden` returns `true` -- plenty of MLP headroom to
spare.

## What the gate checks

`main.cpp` runs six fixed `(latency_cycles, cycles_per_iter, mlp_max)`
triples -- three where MLP headroom is ample, three where the hardware's
`mlp_max` binds first -- and prints the computed distance and hidden
flag for each. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's. A solution that computes
only `required_concurrency` and ignores `mlp_max` entirely (always
`hidden=true`) matches on the three latency-bound cases but is wrong on
all three MLP-bound ones -- believing you can always prefetch far enough
ahead to hide any latency, when the hardware's outstanding-request limit
says otherwise.
