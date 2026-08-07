## Context

A continuous-batching server admits requests into a fixed number of
concurrent slots, $S = $ `max_num_seqs`. Request $i$ becomes eligible for
admission at iteration $\text{arrival}_i$ and needs $\text{gen}_i$ decode
tokens before it is done. Every iteration $t = 0, 1, 2, \dots$:

1. **Admit**: while there is a free slot and a not-yet-admitted request $i$
   with $\text{arrival}_i \le t$, admit the earliest such request — ties in
   $\text{arrival}_i$ broken by original request index (FIFO) — until slots
   run out or no eligible request remains.
2. **Record**: the active set for iteration $t$ is exactly the requests
   occupying a slot at this point (possibly empty, if nothing has arrived
   yet or everything currently running just finished and nothing new is
   eligible yet).
3. **Decode**: every active request advances by one generated token.
4. **Retire**: any request whose generated-token count has now reached
   $\text{gen}_i$ leaves its slot, freeing it for admission at iteration
   $t+1$.

The simulation stops once every request has been both admitted and
retired.

## Task

Implement `simulate_active_set`:

```python
def simulate_active_set(arrival_iters: list[int], gen_lens: list[int], max_num_seqs: int) -> list[list[int]]:
    ...
```

- `arrival_iters` — 1-D integer array, $\text{arrival}_i \ge 0$ for each
  request $i$ (index into the array = original request index, used for the
  FIFO tie-break).
- `gen_lens` — 1-D integer array, $\text{gen}_i \ge 1$, tokens needed
  before request $i$ retires.
- `max_num_seqs` — $S$, the concurrency cap.

Return a list of lists: `result[t]` is the sorted list of request indices
active during iteration $t$, for every iteration from $0$ up to (and
including) the one in which the last request retires.

## Example

```python
arrival_iters = [0, 0, 1]
gen_lens      = [2, 1, 3]
simulate_active_set(arrival_iters, gen_lens, max_num_seqs=2)
# -> [[0, 1], [0, 2], [2], [2]]
```

At $t=0$ requests 0 and 1 both arrive and fill both slots; request 1 (gen
length 1) retires after that iteration. At $t=1$ request 2 (arrived at
$t=1$) backfills the freed slot alongside request 0; request 0 retires
after that iteration (gen length 2). Request 2 alone runs for its
remaining 2 iterations.

## What the gate checks

The fixture holds several hand-picked request pools (serial contention
under $S=1$, two requests that both fit at once, an idle gap while waiting
for a late arrival, staggered FIFO backfill, a same-iteration arrival tie
broken by index) plus several random pools. For each, the grader runs the
same admit/decode/retire simulation and compares the full per-iteration
active-set sequence to yours exactly (`exact_match == 1.0`) — same number
of iterations, same active IDs at every iteration. Admitting out of FIFO
order, decoding before admitting (so a same-step arrival never gets its
first token that step), or stopping the clock during an idle gap instead
of emitting an empty iteration will all diverge from the reference.
