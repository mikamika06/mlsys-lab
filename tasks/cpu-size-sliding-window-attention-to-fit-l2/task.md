## Context

Sliding-window attention only ever attends to the last $W$ tokens, so a
well-tuned kernel keeps that entire window's keys, values, and scores
resident in L2 for the whole computation — no attention-related traffic
spills to DRAM mid-step. For head dimension $D$, $\text{elem\_bytes}$
bytes per key/value element, and $\text{score\_bytes}$ bytes per
attention-score element, one query's total working set is:

$$\text{bytes}(W) = \underbrace{2 D \cdot \text{elem\_bytes}}_{\text{query + output accumulator}} \;+\; W \cdot \underbrace{\left(2 D \cdot \text{elem\_bytes} + \text{score\_bytes}\right)}_{\text{one window slot: K + V + score}}$$

$\text{bytes}(W)$ is strictly increasing in $W$, so there's exactly one
largest window that still fits a given L2 capacity — pick a bigger window
(more context per step) and part of it necessarily evicts to make room,
turning a chunk of what should be pure L2 reuse into traffic all the way
out to a slower level.

## Task

Implement both:

```cpp
long attention_working_set_bytes(int W, int D, int elem_bytes, int score_bytes);
int  choose_max_window(long l2_capacity_bytes, int D, int elem_bytes, int score_bytes);
```

`attention_working_set_bytes` is the formula above, exactly.
`choose_max_window` returns the largest `W >= 0` with
`attention_working_set_bytes(W, ...) <= l2_capacity_bytes`.

## Example

`D=128`, `elem_bytes=4` (fp32), `score_bytes=4`, 256KB L2: the fixed cost
is `2*128*4=1024` bytes, each window slot costs `2*128*4+4=1028` bytes,
and `(262144-1024)/1028 = 254` (floor). `W=254` uses `262136` bytes (fits);
`W=255` would use `263164` bytes (255KB+, doesn't fit).

## What the gate checks

`exact_match` on `(W, ws(W), ws(W+1))` for two fixed scenarios: fp32
K/V/scores at 256KB/head_dim 128 (`W=254`), and bf16 K/V with fp32 scores
at 32KB/head_dim 64 (`W=125`) — mixed element sizes, so a formula that
assumes `elem_bytes == score_bytes` breaks the second scenario even if it
happens to pass the first. In both, `ws(W)` must be `<=` the L2 capacity
and `ws(W+1)` must exceed it; forgetting the fixed Q+output term,
dropping the score term, or an off-by-one in the floor division, shows up
as a wrong `W` or a `ws(W+1)` that doesn't actually overflow.
