## Context

Python distinguishes three kinds of user‑defined callables that produce iterators or awaitable objects:

* **Generator functions** (`def f(): yield ...`) create generator objects.
* **Coroutine functions** (`async def f(): pass`) return coroutine objects when called.
* **Async generator functions** (`async def f(): yield ...`) combine both behaviours.

Internally each function’s code object contains a bitmask `co_flags` that encodes these properties. The relevant flag constants are:

$$
\text{CO_COROUTINE} = 0x2000,\qquad
\text{CO_ASYNC_GENERATOR} = 0x8000,\qquad
\text{CO_GENERATOR} = 0x20.
$$

A function is a coroutine if `co_flags & CO_COROUTINE` is non‑zero, an async generator if `co_flags & CO_ASYNC_GENERATOR`, and a generator if `co_flags & CO_GENERATOR`. Note that an async generator has both the coroutine and async‑generator flags set.

## Task

Implement the function `classify(fn)` that receives any callable object `fn` and returns a list of three integers:

```
[classifier_coroutine, classifier_async_gen, classifier_generator]
```

where each entry is `1` if `fn` has the corresponding property and `0` otherwise. The classification must be based on inspecting `fn.__code__.co_flags`; do **not** use the high‑level helpers from `inspect`.

```python
def classify(fn: Callable) -> List[int]:
    ...
```

The function should work for normal functions, coroutine functions, async generator functions, and generator functions.

## Example

```python
import asyncio

def normal(): pass

async def coro():
    await asyncio.sleep(0)

async def agen():
    yield 1

def gen():
    yield 1

print(classify(normal))   # [0, 0, 0]
print(classify(coro))     # [1, 0, 0]
print(classify(agen))     # [1, 1, 0]
print(classify(gen))      # [0, 0, 1]
```

## What the gate checks

The grader calls `classify` on a set of four functions (normal, coroutine, async generator, generator) and compares the returned list to the reference classification computed from the real CPython code‑object flags. The metric used is **exact_match**; the candidate must return exactly the same integer vectors for all test cases.

The gate will fail if any entry differs or if an exception is raised during evaluation.
