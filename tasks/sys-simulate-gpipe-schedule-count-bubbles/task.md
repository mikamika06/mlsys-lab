## Context

GPipe splits a model into $p$ sequential stages, one per device, and splits
a mini-batch into $m$ microbatches to keep every stage busy. Its schedule is
"fill then drain": every device runs **all** $m$ forward microbatches first,
then **all** $m$ backward microbatches (in reverse microbatch order) — unlike
1F1B schedules that interleave forward and backward. This ordering is
exactly what produces the classic GPipe "bubble" diagram: devices sit idle
while the pipeline fills up at the start and drains at the end.

Assume every forward task and every backward task takes exactly **1**
discrete integer time slot (uniform cost, the standard simplification for
bubble analysis).

## Task

Implement `gpipe_schedule`:

```python
def gpipe_schedule(p: int, m: int) -> dict:
    ...
```

- `p` — number of pipeline stages/devices (`0`-indexed `0..p-1`).
- `m` — number of microbatches (`0`-indexed `0..m-1`).

Each device $i$ executes its $2m$ tasks strictly in this order:
$$
F(0), F(1), \dots, F(m{-}1),\ B(m{-}1), B(m{-}2), \dots, B(0) .
$$
A task starts at the earliest integer time slot such that **all** of the
following already-finished conditions hold:

- **device-order**: that device's previous task in the list above has
  finished (no dependency for a device's very first task, $F(0)$);
- **forward pipeline** (only for $F(j)$ with $i>0$): $F(j)$ on device
  $i-1$ has finished (needs the activation handed off from the previous
  stage);
- **backward pipeline** (only for $B(j)$ with $i<p-1$): $B(j)$ on device
  $i+1$ has finished (needs the gradient handed back from the next stage).

Device $0$'s forwards have no forward-pipeline dependency; device $p-1$'s
backwards have no backward-pipeline dependency.

Return
```python
{"timeline": timeline, "makespan": makespan, "bubble_slots": bubble_slots}
```
where:
- `timeline` — a list of `p` lists, one per device, each a list of `2*m`
  `(start, end)` integer-tuples in that device's execution order
  `[F(0), F(1), ..., F(m-1), B(m-1), ..., B(0)]`, with `end = start + 1`.
- `makespan` — the time slot at which the very last task (on any device)
  finishes: `max` over all tasks' `end`.
- `bubble_slots` — total idle device-slots summed over every device:
  `p * makespan - p * 2 * m` (each device has `2*m` busy slots out of
  `makespan` total slots).

## Example

```python
sched = gpipe_schedule(p=2, m=2)
# device 0: F(0)=[0,1) F(1)=[1,2) B(1)=[4,5) B(0)=[5,6)
# device 1: F(0)=[1,2) F(1)=[2,3) B(1)=[3,4) B(0)=[4,5)
# makespan == 6   (== (m + p - 1) * 2, the standard GPipe result)
# bubble_slots == 2*6 - 2*2*2 == 4
```

## What the gate checks

A single gate, **exact_match**, runs `gpipe_schedule` on several
`(p, m)` configurations and requires your `timeline` (every device's exact
list of `(start, end)` tuples), `makespan`, and `bubble_slots` to all match
a reference simulation built from the dependency rules above, exactly
(everything here is integers — no tolerance is needed or allowed).
