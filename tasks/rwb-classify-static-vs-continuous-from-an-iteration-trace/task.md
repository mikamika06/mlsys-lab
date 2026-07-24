## Context

Two ways an inference server can batch concurrent requests:

- **Request-level (static) batching**: a batch of up to $C$ requests is
  formed, and — because every sequence in the batch runs for as many steps
  as the *longest* member — the whole batch is frozen until every member is
  done, then swapped out wholesale for the next $C$ waiting requests.
- **Iteration-level (continuous) batching**: at every scheduling step, any
  slot whose sequence just finished is immediately backfilled from the
  waiting queue, so a new request can join mid-batch while other members
  are still running.

Given a trace of which sequence IDs were active at each iteration, the two
strategies leave a distinct fingerprint: continuous batching admits a new ID
into the active set *while some previous member is still active*; static
batching never does — its active set only ever changes by being completely
replaced at once (or not at all).

Formally, for iterations $t-1 \to t$ with active sets $A_{t-1}, A_t$, define

$$
\mathrm{new}_t = A_t \setminus A_{t-1}, \qquad
\mathrm{continuing}_t = A_{t-1} \cap A_t .
$$

The run is **continuous** if $\mathrm{new}_t \neq \emptyset$ and
$\mathrm{continuing}_t \neq \emptyset$ for some $t$ (an admission happened
before all current members completed); otherwise it is **static**.

## Task

Implement `classify_scheduling`:

```python
def classify_scheduling(active: np.ndarray) -> str:
    ...
```

- `active` — a 2-D array of shape $(T, N)$ where $T$ is the number of
  iterations in this run and $N$ is the ID universe size; `active[t, i]`
  is truthy iff sequence ID $i$ is active at iteration $t$.

Return the string `"static"` or `"continuous"`.

## Example

```python
import numpy as np

# ids {0,1,2} run together, then swap wholesale to {3,4,5} -> static
static_trace = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
])
classify_scheduling(static_trace)  # -> "static"

# id 2 finishes and id 3 is admitted while 0,1 are still running -> continuous
continuous_trace = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 1],
])
classify_scheduling(continuous_trace)  # -> "continuous"
```

## What the gate checks

The fixture holds several runs recorded from two real scheduler
simulations — one that pads every batch to its slowest member and swaps
wholesale, one that backfills a finished slot from the waiting queue on the
very next step — so each run's true label is known from which simulator
produced it. For every run, the grader re-derives the label from the
`active` matrix alone, using the rule above, and requires

$$
\text{your label} = \text{oracle label}
$$

for every run (`exact_match == 1.0`). A solution that guesses from batch
size, run length, or any signal other than "did an admission ever overlap
with a still-active member" will disagree with the oracle on at least one
run.
