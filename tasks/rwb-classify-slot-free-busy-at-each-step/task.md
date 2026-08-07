## Context

In GPU scheduling and memory-management subsystems (e.g. CUDA stream pools, memory
bank allocators), job assignments to hardware slots are recorded as sparse event
logs.  Each event specifies a time step $t$, a slot index $k$, and a sequence
identifier $i$ (or $i = -1$ to signal a free).  Between events a slot retains
its last known assignment — this is *forward-fill* semantics.

Given $M$ assignment events over $T$ time steps and $K$ slots the goal is to
reconstruct the complete boolean state matrix $B \in \{0,1\}^{T \times K}$ where
$B_{t,k} = 1$ (busy) if slot $k$ holds an active job at step $t$, and $0$ (free)
otherwise.

The per-slot state evolves as

$$
s_{t,k} = \begin{cases}
s_{t-1,k} & \text{no event at step } t \text{ targets slot } k, \\
i         & \text{event assigns } \texttt{seq\_id} = i \geq 0, \\
-1        & \text{event frees the slot } (\texttt{seq\_id} = -1),
\end{cases}
$$

with the boundary condition $s_{-1,k} = -1$ (all slots start free).  The
boolean classification applies the Iverson bracket:

$$
B_{t,k} = [s_{t,k} \neq -1]\,.
$$

## Task

Implement `classify_slots(events, num_steps, num_slots)`:

```python

def classify_slots(events, num_steps, num_slots):
    """
    Args:
        events: list of (step, slot, seq_id) tuples.
                seq_id >= 0 means assigned; seq_id == -1 means freed.
                Multiple events may target the same step (on different slots).
        num_steps: total number of time steps (0 .. num_steps-1).
        num_slots: number of slots (0 .. num_slots-1).

    Returns:
        list[float] of shape (num_steps, num_slots), dtype bool.
        True = busy, False = free.
    """
    ...
```

The function must return a Python boolean array of shape
$(T, K)$.  Slots start free; events are applied in the order given; multiple
events at the same step are all processed before recording that step's row.

## Example

```python
events = [(0, 0, 1), (2, 0, -1), (1, 1, 2)]
result = classify_slots(events, num_steps=4, num_slots=3)
# Step 0: slot 0 = seq 1 (busy), slot 1 = free, slot 2 = free
# Step 1: slot 0 = seq 1 (busy), slot 1 = seq 2 (busy), slot 2 = free
# Step 2: slot 0 freed, slot 1 still seq 2 (busy), slot 2 free
# Step 3: slot 0 free, slot 1 still seq 2 (busy), slot 2 free
# result =
# [[ True, False, False],
#  [ True,  True, False],
#  [False,  True, False],
#  [False,  True, False]]
```

## What the gate checks

The gate uses metric `exact_match`.  Five cases are tested against a reference
oracle that independently replays the event log:

1. **Forward-fill over a gap** — an assignment at step 0 is freed at step 2;
   step 1 must show the slot still busy.
2. **Simultaneous events on different slots** — two assignments at the same
   step.
3. **No events** — the entire matrix must be `False`.
4. **Re-assign, simultaneous free+assign, late free** — exercises the state
   machine across multiple steps with interleaved slot activity.
5. **Rapid reassignment** — a slot is assigned, replaced by a new sequence,
   then freed, then re-assigned in consecutive steps.

A correct forward-fill loop that processes every event and records the state at
every step will pass; implementations that skip idle steps, ignore the $-1$ free
sentinel, or mutate state in-place without a proper snapshot will fail.
