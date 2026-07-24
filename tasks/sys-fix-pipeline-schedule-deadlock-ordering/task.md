## Context

Pipeline parallelism splits a model into sequential stages and executes multiple microbatches at the same time. Each stage has dependencies on neighboring stages.

For a microbatch $m$ and stage $s$, a forward operation $F(s,m)$ depends on the previous stage:

$$
F(s,m) \leftarrow F(s-1,m)
$$

when $s > 0$. A backward operation $B(s,m)$ depends on its own forward result and the next stage's backward result:

$$
B(s,m) \leftarrow F(s,m), B(s+1,m)
$$

when $s < S-1$.

A scheduler must preserve these dependencies while keeping all stages busy. A bad dependency ordering can cause a pipeline to stall or execute an invalid backward pass before its inputs exist.

## Task

Implement `pipeline_schedule(stages, microbatches)`.

The function must return a list of pipeline clock cycles. Each cycle is a list with one entry per stage. An entry is either `None` or a tuple:

```python
(stage, microbatch, phase)
```

where `phase` is either `"F"` for forward or `"B"` for backward.

The schedule must follow this deterministic policy:

1. At every cycle, each stage may run at most one operation.
2. An operation is eligible only when all of its dependencies have already completed.
3. When multiple operations are eligible for the same stage, prefer backward operations over forward operations.
4. For operations with the same phase priority, choose the smaller microbatch index.

The schedule ends when every forward and backward operation has completed.

## Example

For:

```python
pipeline_schedule(2, 2)
```

one valid returned shape is:

```python
[
    [(0, 0, "F"), None],
    [(0, 1, "F"), (1, 0, "F")],
    [(0, 0, "B"), (1, 1, "F")],
    [(0, 1, "B"), (1, 0, "B")],
    [None, (1, 1, "B")]
]
```

The exact output must follow the priority rules above.

## What the gate checks

The gate builds the expected schedule with an independent dependency simulator. It checks that the returned schedule is exactly the oracle schedule, including operation order.

A schedule that leaves unfinished microbatches, violates dependencies, or uses a different ordering policy fails the gate.
