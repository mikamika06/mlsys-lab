## Context

A Python generator is an iterator whose execution can be paused and resumed. The
CPython runtime tracks a generator's lifecycle with states exposed by
`inspect.getgeneratorstate`.

A generator can be in one of these states:

- `GEN_CREATED`: the generator object exists but its body has not started.
- `GEN_RUNNING`: the generator body is currently executing.
- `GEN_SUSPENDED`: the generator has yielded a value and is paused.
- `GEN_CLOSED`: the generator has finished or has been closed.

The state transition sequence depends on when execution is observed. For a
generator object $g$, creating it does not execute the function body, while
calling `next(g)` starts execution until the next `yield` point.

## Task

Implement `generator_states()`.

The function must create a small scripted generator and return a list of state
names observed during its lifecycle. The returned list must include observations
at:

1. immediately after generator creation,
2. while the generator body is executing before its first `yield`,
3. after the first `yield` when execution is paused,
4. after the second `yield` when execution is paused,
5. after the generator has completed.

Return the names exactly as produced by `inspect.getgeneratorstate`, such as
`"GEN_CREATED"` and `"GEN_SUSPENDED"`.

The function signature is:

```python
def generator_states():
    ...
```

Do not use hardcoded state lists. The observations must come from inspecting the
actual generator object.

## Example

```python
states = generator_states()

# Example shape:
# [
#   "GEN_CREATED",
#   "GEN_RUNNING",
#   "GEN_SUSPENDED",
#   "GEN_SUSPENDED",
#   "GEN_CLOSED",
# ]
```

## What the gate checks

The gate builds the same scripted generator with real CPython
`inspect.getgeneratorstate` calls and uses it as the reference behavior.

The returned list is compared with the oracle result using exact equality.
Solutions that guess the sequence without observing generator state transitions,
or that omit the `GEN_RUNNING` checkpoint, will fail.
