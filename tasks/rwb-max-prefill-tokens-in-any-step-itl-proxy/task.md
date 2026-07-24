## Context

In continuous batching, every iteration must first run one decode step for
every already-running request (it can't be delayed — that request is
mid-generation) before any *new* prompt can be prefilled. **Unchunked**
prefill schedules a whole new prompt in a single iteration, however long
it is: a 4000-token prompt means one iteration does 4000 tokens of
prefill work, during which every other request's decode step is delayed
behind it — a huge spike in inter-token latency (ITL) for everyone else.

**Chunked prefill** caps how many tokens (decode + prefill combined) an
iteration is allowed to process, a fixed budget $C$. Each iteration first
spends $d_t$ tokens on whatever decode requests are already running, then
spends the rest of the budget, $\max(C - d_t,\, 0)$, on prefill — taking
tokens off the front of the pending-prompt queue, splitting a large prompt
across as many iterations as it takes:

$$
\text{prefill\_tokens}(t) = \min\!\Big(\text{tokens left in queue reachable this step},\; \max(C - d_t,\, 0)\Big)
$$

The value that matters for worst-case ITL is not the total prefill work,
but the **single worst iteration**: $\max_t \text{prefill\_tokens}(t)$.
Chunking bounds this by (roughly) $C$; without it, the bound is the
largest prompt in the workload.

## Task

Implement `prefill_chunking_max_step(prompt_lengths, decode_load, budget)`:

```python
def prefill_chunking_max_step(prompt_lengths: list[int], decode_load: list[int], budget: int):
    ...
```

- `prompt_lengths`: a FIFO queue of pending prefill jobs (their prompt
  token counts).
- `decode_load[t]`: tokens that already-running decode requests
  unconditionally consume at iteration $t$ (use $0$ for any iteration
  past the end of this list).
- `budget`: max total tokens (decode + prefill) an iteration may process
  under the **chunked** policy.

Simulate the scheduler under both policies and return
`(max_prefill_tokens_chunked, max_prefill_tokens_unchunked)`:

- **Chunked**: run iterations until the prefill queue is empty. Each
  iteration, `decode_load[t]` tokens run first (unconditionally); the
  remaining budget is spent taking (possibly partial) tokens off the
  front of the queue, packing multiple small jobs into the same
  iteration if they fit. Track the max prefill tokens processed in any
  one iteration.
- **Unchunked**: every queued job is prefilled to completion in one
  single iteration with no cap, so the worst iteration is simply the
  largest prompt in the workload.

## Example

```python
prefill_chunking_max_step([500, 20, 700], decode_load=[0]*10, budget=128)
# chunked: budget is never reduced by decode (all zero), so every
#   iteration processes exactly 128 prefill tokens until the queue drains
#   -> max = 128
# unchunked: the whole 700-token prompt runs in one iteration -> max = 700
# -> (128, 700)
```

## What the gate checks

The gate runs several hand-built workloads (a clean budget-bound case like
the example, decode load that eats into the budget every iteration, an
empty queue, a single prompt smaller than the budget so chunking changes
nothing, several small prompts packed together in one chunked iteration,
and an extreme tiny-budget case) plus randomly generated `(prompt_lengths,
decode_load, budget)` triples from a seeded generator.

For every case the reference runs both simulations exactly as described
— FIFO, partial-job packing under chunking, decode load consumed first —
and returns the two maxima. Your `(chunked, unchunked)` pair is compared
to it with exact equality. A solution that computes the chunked max as
simply `min(budget, max(prompt_lengths))`, ignoring that `decode_load`
can shrink the usable budget on some iterations, will match whenever
decode load happens to be zero but disagree as soon as a case has
nonzero decode load competing for the same budget.
