## Context

DeepSpeed-FastGen's **Dynamic SplitFuse** scheduler fills every
iteration's fixed token budget $C$ (`max_num_batched_tokens`) with two
kinds of work:

- **Decode**: every currently running sequence contributes exactly one
  new token this iteration. Decode always gets priority and is never
  chunked.
- **Prefill**: the remaining budget $C - R$ (where $R$ is the number of
  running sequences) is spent greedily on the FIFO queue of pending
  prompts. A prompt that doesn't fully fit in the remaining budget gets
  **chunked** — only part of it is prefilled this iteration, and the
  rest waits for future iterations. If a prompt finishes prefilling with
  budget still left over, the scheduler immediately moves on to the next
  queued prompt in the *same* iteration.

A prompt that finishes prefilling during iteration $t$ joins the running
(decoding) set starting iteration $t+1$ — it contributes to $R$, and
therefore eats into the budget, from then on.

For iteration $t$ with running count $R_t$ and remaining pending queue
$Q_t$ (a list of remaining-to-prefill lengths, in FIFO order):

$$
\text{decode\_tokens}_t = R_t, \qquad
\text{prefill\_chunk\_tokens}_t = \min\!\Big(C - R_t,\ \textstyle\sum \text{as much of } Q_t \text{ as fits}\Big).
$$

## Task

Implement `dynamic_splitfuse_pack`:

```python
def dynamic_splitfuse_pack(initial_running: int, pending: list[int], C: int) -> list[tuple[int, int]]:
    ...
```

* `initial_running` — number of decode sequences already running before
  iteration 1.
* `pending` — FIFO queue of prefill request prompt lengths, not yet
  started.
* `C` — `max_num_batched_tokens`, the total token budget per iteration.

Simulate iterations until `pending` is fully drained (every prompt fully
prefilled), seating decode tokens first and then greedily chunking
pending prefills into whatever budget is left over, moving to the next
queued prompt within the same iteration whenever one finishes with
budget to spare. A prompt that finishes prefilling in an iteration joins
the running set starting the *next* iteration.

Return the list of `(decode_tokens, prefill_chunk_tokens)` pairs, one
entry per iteration, in order.

## Example

```python
dynamic_splitfuse_pack(initial_running=0, pending=[8, 4], C=8)
# iter 1: decode=0 (nothing running yet), budget=8 -> consumes all 8
#         tokens of the first prompt, finishing it exactly, budget=0
#         left, so the second prompt doesn't start yet.
#         -> (0, 8)
# iter 2: decode=1 (first prompt now running), budget=7 -> the second
#         prompt (4 tokens) fits entirely with room to spare, but the
#         queue is now empty.
#         -> (1, 4)
# result: [(0, 8), (1, 4)]
```

## What the gate checks

The gate, **exact_match**, runs your function against a reference
simulation on several hand-picked cases (including a case where one
prompt finishes and the next starts within the same iteration) plus a
handful of deterministically generated cases. It compares the full
returned list of `(decode_tokens, prefill_chunk_tokens)` pairs
element-by-element, including the total number of iterations — any
mismatch in a single iteration's split, or a wrong iteration count,
fails that case.
