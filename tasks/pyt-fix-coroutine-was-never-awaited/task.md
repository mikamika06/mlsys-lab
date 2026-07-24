## Context

When a coroutine function `async def f(...)` is called, Python returns a
**coroutine object** $c$ in state CREATED.  The coroutine only begins
executing when the event loop drives it, which happens when $c$ appears
on the right-hand side of `await` (or is passed to `asyncio.run`,
`asyncio.gather`, etc.).

A coroutine object $c$ transitions through states

$$S(c) \in \{\text{CREATED},\;\text{RUNNING},\;\text{SUSPENDED},\;\text{FINISHED}\}.$$

Without `await`, $S(c)$ never leaves CREATED and the coroutine's return
value is never produced.

In a pipeline that chains two async steps — squaring then incrementing
— the correct call sequence for input $n$ is

$$v_1 = \texttt{await}\;\texttt{delayed\_square}(n), \qquad
  v_2 = \texttt{await}\;\texttt{delayed\_increment}(v_1),$$

producing $v_2 = n^2 + 1$.  Omitting `await` yields coroutine objects
instead of integers, and the downstream function receives a coroutine
object as its argument rather than a numeric value.

## Task

The file `starter.py` contains `collect_results(numbers)`, an `async`
function that should process a list of integers through a two-stage
async pipeline (square, then increment) and return the list of final
values.  However, it has a bug: the `await` keyword is missing, so
every intermediate value is a coroutine object rather than an integer.

Fix the code so that every coroutine is actually driven to completion.
Do **not** change the helper functions `delayed_square` or
`delayed_increment`.  The fixed function must remain `async` and return
a plain `list[int]` when awaited through an event loop.

## Example

```python
import asyncio
from starter import collect_results

result = asyncio.run(collect_results([2, 3]))
# Expected: [5, 10]   because 2^2+1=5, 3^2+1=10
```

## What the gate checks

The gate runs `collect_results` inside `asyncio.run` on several input
lists and compares the returned list with a reference computed through
the same async pipeline.  A list of coroutine objects (the broken
behaviour) does not equal a list of integers, so `exact_match` is 0.0
until every `await` is in place.
