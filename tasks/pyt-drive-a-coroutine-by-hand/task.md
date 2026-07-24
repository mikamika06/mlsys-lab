## Context

An `async def` function creates a coroutine object. An event loop normally drives that coroutine, but the coroutine protocol can also be used directly.

Calling `send(None)` starts execution. When the coroutine reaches an `await`, Python runs the awaitable's `__await__()` iterator. If that iterator yields a value, the coroutine itself yields that value to its caller. Sending another value resumes the coroutine until it yields again or finishes.

For a coroutine with yielded sentinels $s_1, s_2, \dots, s_k$ and final return value $r$, manual driving collects the sequence

$$
(s_1, s_2, \dots, s_k, r).
$$

This task uses custom awaitables that yield deterministic sentinel objects. No event loop is involved.

## Task

Implement `drive_coroutine(coro_factory)`.

The argument `coro_factory` is a zero-argument callable that returns a fresh coroutine object. Drive that coroutine manually using `.send(None)` until it finishes.

Return a tuple:

```python
(yielded_values, result)
```

where:

- `yielded_values` is a list containing every value produced by the coroutine while it is suspended at an `await`.
- `result` is the final value returned by the coroutine.

Do not use `asyncio` or create an event loop.

## Example

```python
class MarkerAwaitable:
    def __init__(self, marker):
        self.marker = marker

    def __await__(self):
        received = yield self.marker
        return received + 1


async def sample():
    a = await MarkerAwaitable(10)
    b = await MarkerAwaitable(20)
    return a + b


yielded, result = drive_coroutine(sample)
# yielded == [10, 20]
# result == 3
```

## What the gate checks

The gate builds several coroutines using real Python coroutine objects and custom awaitables. It computes the expected output by manually following the coroutine protocol with `.send(None)` and compares the submitted implementation against that oracle.

The returned yielded sequence and final result must match exactly.
