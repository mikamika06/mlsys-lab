## Context

A serving engine processes $N$ requests' prompt (prefill) tokens under a
fixed per-iteration token budget $C$, in strict FIFO arrival order. Two
scheduling policies handle the budget differently:

- **Monolithic** (no chunked prefill): a request's prompt cannot be
  split across iterations. Each iteration greedily packs whole prompts
  from the front of the queue while they still fit in the remaining
  budget, and stops at the first one that doesn't (deferring it and
  everyone behind it — no reordering). A single prompt longer than $C$
  still has to run, alone, in one (over-budget) iteration, since there
  is no way to shrink it.
- **Chunked prefill**: a prompt *may* be split across iteration
  boundaries. Each iteration consumes exactly up to $C$ tokens from the
  front of the queue — finishing the head request and moving to the
  next if budget remains, or leaving it partially done to resume first
  next iteration if the budget runs out mid-prompt. This never wastes
  budget.

A request's **TTFT iteration** is the iteration at which its prefill
*completes* (all of its prompt tokens have been processed) — that's when
its first output token can be produced. Because chunking packs the
budget without waste, it can finish `iters_chunked` $= \lceil \sum_i
\ell_i / C \rceil$ total iterations, and it can also change *individual*
requests' TTFT relative to monolithic (usually earlier for requests
queued behind one very long prompt, since a long prompt is chopped into
pieces that share iterations with others instead of hogging one alone).

## Task

Implement `compare_chunked_vs_monolithic`:

```python
def compare_chunked_vs_monolithic(prompt_lens: list[int], C: int) -> dict:
    ...
```

- `prompt_lens`: list of $N$ positive ints, prompt length per request,
  in FIFO arrival order.
- `C`: positive int, the per-iteration token budget.
- Simulate both policies exactly as described above.

Return a dict:
- `"iters_mono"`, `"iters_chunked"`: total iterations to finish all
  requests, under each policy.
- `"ttft_mono"`, `"ttft_chunked"`: length-`N` lists of the 1-indexed
  iteration at which each request's prefill completes, under each
  policy.

## Example

```python
prompt_lens = [10, 3, 25, 2]
C = 12

out = compare_chunked_vs_monolithic(prompt_lens, C)
# Monolithic: iter 1 packs [10] (3 doesn't fit in the remaining 2) ->
#   iter 2 packs [3] alone (25 doesn't fit) -> iter 3 runs [25] alone
#   (over budget, unsplittable) -> iter 4 packs [2].
#   iters_mono == 4, ttft_mono == [1, 2, 3, 4]
# Chunked: iter 1 takes 10 from req0 (done) + 2 from req1 (2 left) ->
#   iter 2 takes 1 from req1 (done) + 11 from req2 (14 left) ->
#   iter 3 takes 12 from req2 (2 left) -> iter 4 takes 2 from req2 (done)
#   + 2 from req3 (done). iters_chunked == 4, ttft_chunked == [1, 2, 4, 4]
```

## What the gate checks

The grader loads a committed fixture — a realistic skewed workload (a
mix of short prompts and two long ones, 200 and 130 tokens, under
budget 64) — plus several additional seeded random workloads, and
replays the exact same two-policy simulation independently in Python
(never calling your function, never hardcoding an expected value).

`exact_match` is the fraction of all checked values — both iteration
counts and every entry of both TTFT lists, across every case — that
match the oracle exactly. The gate requires `1.0`. Allowing monolithic
to reorder or skip a request that doesn't fit, letting chunked waste
budget instead of always taking `min(remaining, budget)`, or an
off-by-one in when an iteration counter increments will all show up as a
mismatch somewhere in the TTFT lists or iteration totals.
