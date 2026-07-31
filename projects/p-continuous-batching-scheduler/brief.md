# A scheduler that keeps its promise

Our inference service serves requests one at a time. The queue is growing, the
card sits idle between requests, and at peak load users see latencies measured
in minutes. Product promised p95 under two seconds and 40 requests per second;
right now we're hitting neither.

The fix is known: serve requests together, splicing new ones into an
already-running batch, and keep KV in blocks instead of one contiguous chunk
sized for the longest possible output. Someone has to write it.

We can't just pull in a ready-made engine — we need to understand the
mechanics ourselves, because we'll be tuning prod around it next.

## What you write

Three files in `sched/`. The rest of the skeleton is harness and tests — you
can change those too.

### `sched/allocator.py`

```python
class Allocator:
    def __init__(self, num_blocks: int, block_size: int): ...
    def allocate(self) -> int          # lowest free block number, ref=1
    def share(self, block: int) -> int  # ref += 1, returns the same number
    def release(self, block: int) -> None   # ref -= 1; block frees at zero
    def free_count(self) -> int
    def register(self, block: int, key: str) -> None
    def lookup(self, key: str) -> int | None
```

A block only counts as free once its refcount hits zero. A second `release`
after it's already at zero must not free the block a second time.

### `sched/policy.py`

```python
def victim(running: list) -> object      # who gets preempted
def should_admit(state: dict) -> bool    # whether to admit the next request
```

The convention the grader checks against: the sequence with the most tokens
(`prompt_len + decoded`) gets preempted; ties go to the higher `rid`.
`should_admit` receives `{"running", "max_seqs", "free_blocks", "blocks_needed"}`
and is the **only** place that decides the concurrent-sequence limit. The
scheduler must not duplicate this check internally — otherwise the policy
can't be swapped out or tested in isolation.

A preempted sequence under `recompute` loses all progress and goes back to
the front of the queue. After `max_preemptions` preemptions it's considered
unschedulable.

### `sched/scheduler.py`

```python
class Scheduler:
    def __init__(self, config: dict): ...
    def add(self, requests: list[dict]) -> None
    def step(self) -> dict
    def run(self, max_steps: int = 100000) -> dict
```

A request is `{"rid": str, "arrival": int, "prompt": list[int], "output_len": int}`.

`config` holds: `block_size`, `num_blocks`, `max_batch_tokens`, `max_seqs`,
`chunked_prefill`, `prefix_cache`, `preemption` (`"recompute"` or `"swap"`),
`swap_blocks`, `max_preemptions`, `prefill_cost`, `decode_cost`, `step_overhead`.

`step()` returns `{"t", "prefill_tokens", "decode_tokens", "running", "blocks_used", "ids"}`,
where `ids` is a tuple of the request ids that got work done on this step, in
execution order: prefilled ones first, then decoded ones.

`run()` returns metrics: `finished`, `rejected`, `preemptions`, `steps`,
`prefill_tokens`, `decode_tokens`, `ttft_p50`, `latency_p95`, `throughput`,
`cache_hit_rate`.

## Timing rules

A step costs `step_overhead + prefill_tokens * prefill_cost + decode_tokens * decode_cost`.
The clock is integer. No wall-clock time — otherwise the result isn't
reproducible.

## Step order of operations

1. Admit anyone with enough blocks who doesn't push past `max_seqs`.
2. Prefill: give each not-yet-prefilled sequence a chunk within the step's
   shared token budget. Without `chunked_prefill` — it's the whole prompt or
   nothing.
3. Decode: give each prefilled sequence one token, if blocks and budget allow.
4. If blocks run short, preempt someone. If only one sequence is left running
   and it still doesn't fit, it's unschedulable: mark it `rejected`.

Progress is mandatory. A configuration where nobody ever finishes is a
scheduler defect, not a property of the workload.

## How this gets checked

The grader computes the reference itself, on the same traces. Metrics and the
step sequence are compared. The last milestone is different: you write a test
that catches a broken policy, and we deliberately break the policy to see if
your test notices.

```
mlsys project start p-continuous-batching-scheduler
mlsys project grade p-continuous-batching-scheduler --milestone 1
```
