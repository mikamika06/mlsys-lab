## Context

Continuous batching (iteration-level scheduling), as used by inference
servers like vLLM, re-decides the running batch at *every* decode step
instead of waiting for a fixed batch to fully finish. A request can be
**admitted** into the running batch, keep running, get **preempted**
(evicted, to resume later) to make room for a higher-priority request, or
finish and leave for good.

Model a fixed-capacity, priority-preemptive scheduler. There are $n$
requests; request $i$ has an arrival step $a_i$, a total required number
of running-steps $\ell_i$, and a priority $p_i$ (**lower** $p_i$ =
**higher** priority; ties broken by request id, lower id wins).

At every step $t = 0, \dots, \text{num\_steps}-1$, the running set is
**recomputed from scratch**: among every request that has arrived
($a_i \le t$) and has not yet finished (its remaining running-steps
counter is still $> 0$), take the `budget` best-priority ones (sorted by
$(p_i, i)$ ascending). Everyone selected then makes one step of progress
(their remaining counter decreases by 1); if it reaches $0$ they are
finished and drop out of consideration for good.

$$
\text{running}_t = \text{first \texttt{budget} of } \big\{\, i : a_i \le t,\ \text{remaining}_i > 0 \,\big\}, \text{ sorted by } (p_i, i)
$$

## Task

Implement `schedule_admit_preempt`:

```python
def schedule_admit_preempt(arrivals, lengths, priorities, budget, num_steps):
    ...
```

- `arrivals`, `lengths`, `priorities`: same-length lists of non-negative
  ints, one entry per request (request id = its index into these lists).
- `budget`: max number of requests that can be in the running set at
  once.
- `num_steps`: number of scheduling steps to simulate, starting at $t=0$.

Return a list of length `num_steps`. Step `t`'s entry is a dict:

```python
{"admitted": [...], "preempted": [...]}
```

- `"admitted"`: sorted request ids in $\text{running}_t$ that were **not**
  in $\text{running}_{t-1}$ (for $t=0$, everyone in $\text{running}_0$
  counts as admitted).
- `"preempted"`: sorted request ids that were in $\text{running}_{t-1}$,
  still have remaining work (they didn't just finish), but are **not** in
  $\text{running}_t$.

A request that simply finishes (its remaining counter hits `0`) is
neither admitted nor preempted when it drops out — it just leaves.

## Example

```python
arrivals   = [0, 0, 1]
lengths    = [3, 2, 4]
priorities = [1, 0, 2]   # request 1 is highest priority, request 2 lowest
budget     = 2

schedule_admit_preempt(arrivals, lengths, priorities, budget, num_steps=3)
```

- `t=0`: candidates `{0, 1}` (2 hasn't arrived yet). Both fit in budget 2
  -> running = `{0, 1}` -> `admitted=[0, 1]`, `preempted=[]`.
- `t=1`: candidates `{0, 1, 2}` (all arrived, all still have work).
  Sorted by priority: `1 (p=0), 0 (p=1), 2 (p=2)` -> top 2 = `{1, 0}` ->
  running unchanged -> `admitted=[]`, `preempted=[]`.
- `t=2`: request 1 finished after step 1 (`length=2`), so candidates are
  `{0, 2}` -> running = `{0, 2}` -> `admitted=[2]`, `preempted=[]` (1 isn't
  "preempted", it finished).

## What the gate checks

The grader runs several seeded `(arrivals, lengths, priorities, budget,
num_steps)` scenarios — including one deliberately built so a
later-arriving, higher-priority request must preempt an already-running,
lower-priority one — through a straightforward simulation of the rule
above, and compares your `admitted`/`preempted` lists step by step.

`exact_match` is `1.0` only if every step's `admitted` and `preempted`
lists match the reference exactly, across every scenario (must equal
`1.0`). Recomputing the running set incrementally instead of from
scratch, breaking priority ties the wrong way, or mislabeling a finished
request as "preempted" will all cause a mismatch somewhere in the
simulation and fail this gate.
