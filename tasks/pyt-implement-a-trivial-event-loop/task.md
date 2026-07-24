## Context

Python coroutines can suspend execution and later continue from the same point.
A minimal event loop only needs a queue of runnable generators and a way to
resume a generator after an awaited value becomes available.

In this task, a `Future` is an awaitable-like object that completes after a
fixed number of scheduler ticks. When a coroutine yields a future, the event
loop stores the coroutine until the future is ready, then sends the future's
value back into the coroutine.

For a set of tasks, the scheduler returns a completion sequence

$$
C = (c_1, c_2, \dots, c_n)
$$

where each $c_i$ is the value returned by a coroutine at the moment it finishes.

## Task

Implement `Future` and `run_event_loop(tasks)`.

The required interface is:

```python
class Future:
    def __init__(self, delay, value):
        ...

    def result(self):
        ...


def run_event_loop(tasks):
    ...
```

`tasks` is a list of generator coroutines. A coroutine may yield one or more
`Future` objects and must eventually return a completion string.

`Future(delay, value)` becomes available after `delay` event-loop ticks.
When a future is ready, the event loop must resume the waiting coroutine by
sending `future.result()` into it.

`run_event_loop(tasks)` must return completion values in the exact order in
which tasks finish. Do not use `asyncio`.

## Example

```python
def worker(name, future):
    value = yield future
    return name + ":" + value

tasks = [
    worker("fast", Future(1, "ok")),
    worker("slow", Future(3, "ok")),
]

run_event_loop(tasks)
# ["fast:ok", "slow:ok"]
```

## What the gate checks

The gate builds several coroutine workloads and compares the result against an
independent event-loop implementation that follows the same coroutine
semantics. The returned completion order must exactly match the computed
reference result.

The `exact_match` score is `1.0` only when all workloads pass.
