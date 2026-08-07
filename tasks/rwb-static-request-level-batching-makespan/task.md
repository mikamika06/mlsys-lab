## Context

**Static (request-level) batching** is the simplest way to serve a queue
of generation requests: group up to $S$ requests into a batch, run the
whole batch forward one decode step at a time, and only start the NEXT
batch once every request in the CURRENT batch has produced its last
token. Because every request in a batch shares the same iteration clock,
a request that only needs a few tokens still occupies its batch slot
(idling) until the slowest request in that same batch finishes — this
**batch-blocking** behavior is exactly why continuous batching (which
lets finished requests exit and new ones join mid-batch) was invented to
replace it.

For a batch $B$ of requests with output lengths $\{\ell_1,\dots,\ell_k\}$
($k \le S$), the batch takes $\max_i \ell_i$ iterations, and the total
time to drain the whole queue (the **makespan**) is the sum over all
batches:

$$
\text{makespan} = \sum_{\text{batch } B} \max_{i \in B} \ell_i .
$$

## Task

Implement `static_batching_makespan`:

```python
def static_batching_makespan(output_lens: list[int], batch_size: int):
    ...
```

* `output_lens` — 1-D int array, the number of decode iterations each
  queued request needs, in arrival (queue) order.
* `batch_size` — $S$, the max number of requests processed together in one
  static batch.

Form batches by taking the next up-to-`batch_size` requests from the queue
in order (the last batch may be smaller). Return
`(makespan, batch_iter_counts)`:

* `makespan` — total iterations across every batch (a plain Python `int`).
* `batch_iter_counts` — a list of ints, one per batch, in order: that
  batch's iteration count (its `max(output_lens)`).

## Example

```python

output_lens = [3, 10, 4, 2, 7]
makespan, counts = static_batching_makespan(output_lens, batch_size=2)
# batch 0: [3, 10]  -> 10 iterations (the 3-token request idles for 7 of them)
# batch 1: [4, 2]   ->  4 iterations
# batch 2: [7]      ->  7 iterations
# counts = [10, 4, 7], makespan = 21
```

## What the gate checks

A single gate, **exact_match**, compares your `(makespan,
batch_iter_counts)` against a reference computed the same way on a fixed
fixture (`output_lens.npy`, 23 requests) with `batch_size = 4`. Both the
total makespan and every per-batch count must match exactly.
