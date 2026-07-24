## Context

Pipeline parallel training splits a model into stages. Each stage executes forward
and backward operations for multiple microbatches. A naive schedule executes all
forward operations first and then all backward operations, leaving pipeline stages
idle.

The 1F1B (one-forward-one-backward) schedule reduces pipeline bubbles by starting
backward work as soon as dependencies allow it. For a pipeline with $S$ stages,
stage $i$ has a warmup period of

$$w_i = S - i - 1$$

forward operations before it can begin alternating forward and backward work.

A schedule is represented by the ordered operations executed by each stage. The
operation count per stage is $2M$, where $M$ is the number of microbatches. A
forward operation is written as `F{k}` and a backward operation is written as
`B{k}` for microbatch $k$.

## Task

Implement `generate_1f1b_schedule(stages, microbatches)`:

```python
def generate_1f1b_schedule(stages: int, microbatches: int) -> list[list[str]]:
    ...
```

Return a list with one entry per pipeline stage. Each entry is the local execution
order of that stage using the 1F1B schedule.

Stages are indexed from $0$ to $S-1$. For stage $i$:

1. Execute the warmup forwards `F0`, `F1`, ..., up to `F(w_i - 1)`.
2. Repeatedly issue the next forward operation while available, and issue a
   backward operation whenever the number of completed forwards exceeds the
   warmup amount.
3. After all forwards are issued, finish remaining backward operations in order.

The function should support positive integer values of `stages` and
`microbatches`.

## Example

```python
schedule = generate_1f1b_schedule(3, 4)

# [
#   ["F0", "F1", "F2", "F3", "B0", "B1", "B2", "B3"],
#   ["F0", "F1", "F2", "B0", "F3", "B1", "B2", "B3"],
#   ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3"]
# ]
```

## What the gate checks

The gate computes the expected schedule using an independent schedule generator
and compares the returned stage operation lists exactly. The comparison requires
the stage ordering and every operation token to match the oracle result.
