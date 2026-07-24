## Context

During model training, memory usage changes as operations allocate and release tensors. A memory timeline can be reconstructed by replaying allocation and free events in execution order.

For a sequence of events, let $L_t$ be the live memory after step $t$. An allocation event with size $s$ changes memory by

$$
L_t = L_{t-1} + s,
$$

while a free event changes memory by

$$
L_t = L_{t-1} - s.
$$

The peak memory is the maximum live memory observed over all steps:

$$
P = \max_t L_t .
$$

The corresponding step index is the first step where this maximum is reached.

Each event is represented as a tuple `(kind, bytes)` where `kind` is either `"alloc"` or `"free"`. The byte count is a non-negative integer.

## Task

Implement `peak_memory_timeline(events)`:

```python
def peak_memory_timeline(events):
    ...
```

The function receives a list of memory events and returns a tuple:

```python
(peak_bytes, peak_step)
```

where:

- `peak_bytes` is the maximum number of live bytes during the replay.
- `peak_step` is the zero-based event index where that maximum first occurs.

The input sequence is valid: frees never exceed currently live memory.

## Example

```python
events = [
    ("alloc", 400),
    ("alloc", 900),
    ("free", 300),
    ("alloc", 200),
]

peak_memory_timeline(events)
# (1300, 1)
```

After each step the live bytes are $400$, $1300$, $1000$, and $1200$, so the peak is $1300$ at step $1$.

## What the gate checks

The gate generates several allocation/free timelines and computes the expected result by independently replaying the events and tracking the maximum live byte count. Your implementation must return the exact `(peak_bytes, peak_step)` pair for every case.
