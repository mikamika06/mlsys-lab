## Context

A Python statement such as `counter += 1` is not one indivisible operation. A simplified model of a non-atomic increment contains three steps:

$$
\mathrm{LOAD}: r \leftarrow c,\qquad
\mathrm{ADD}: r \leftarrow r+1,\qquad
\mathrm{STORE}: c \leftarrow r .
$$

If multiple threads interleave these steps, one thread can overwrite another thread's update. The Global Interpreter Lock does not make a multi-step operation like this behave as a mathematical atomic increment.

For $N$ threads where each thread performs $M$ increments, there are $N \times M$ logical increments. The final counter value depends on the ordering of the individual `LOAD`, `ADD`, and `STORE` operations. The goal is to reconstruct the set of all reachable final values under this interleaving model.

A thread cannot start its next increment until its current increment has completed. A thread's register value is private and is only changed by its own `LOAD` and `ADD` steps.

## Task

Implement `race_outcomes(n_threads, n_increments)`:

```python
def race_outcomes(n_threads: int, n_increments: int) -> set[int]:
    ...
```

Return the set of all possible final values of the shared counter. Assume the initial counter value is $0$.

The function should model all valid interleavings of the three-step increment operation. It should not use threads, sleeps, or timing assumptions.

## Example

```python
print(race_outcomes(2, 1))
# {1}
```

Two threads each execute one non-atomic increment. Both can load $0$, and the later store overwrites the earlier store, so the only possible final value is $1$.

## What the gate checks

The gate compares the returned set against a reference simulator that explores the reachable states of the interleaving model. The simulator computes possible outcomes from the transition rules directly rather than using a list of expected answers.

The tested cases include multiple threads and multiple increments. Returning only the minimum and maximum values is insufficient because some intermediate values may be unreachable.
