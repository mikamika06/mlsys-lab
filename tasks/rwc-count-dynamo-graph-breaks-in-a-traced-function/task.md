## Context

When a Python function is traced by a just‑in‑time compiler such as PyTorch’s Dynamo, the tracer records every operation it encounters.  
Some operations cannot be represented in a static graph (e.g., calls to `print`, data‑dependent branches, or unsupported builtins).  Each time such an event occurs the current subgraph is *broken* and a new one starts.

Let

- $B$ denote the number of break events that force a new subgraph.
- $S$ denote the number of contiguous sequences of normal operations between breaks.

Then $S$ counts every maximal block of non‑break events that appears in the trace.

## Task

Implement `count_breaks_and_subgraphs(events)`:

```python
def count_breaks_and_subgraphs(events: list[str]) -> tuple[int, int]:
    ...
```

`events` is a list of strings describing traced operations.  
An event string that starts with `"break_"` (case‑sensitive) signals a graph break; all other events are considered normal operations.

The function must return a two‑tuple `(B, S)` where:

- `B` is the number of break events in `events`.
- `S` is the number of contiguous blocks of non‑break events.

Both values should be plain Python integers.  The implementation must not use any external libraries beyond the standard library.

## Example

```python
>>> ev = ["op1", "op2", "break_unsupported", "op3", "break_branch", "op4"]
>>> count_breaks_and_subgraphs(ev)
(2, 3)
```

Here there are two break events (`"break_unsupported"` and `"break_branch"`).  
The non‑break blocks are `["op1","op2"]`, `["op3"]`, and `["op4"]`, so $S = 3$.

## What the gate checks

A single gate named `exact_match` compares the tuple returned by your function with a reference computed by the grader.  
The comparison is strict equality; any mismatch or exception causes the gate to fail.  No other metrics are evaluated.
