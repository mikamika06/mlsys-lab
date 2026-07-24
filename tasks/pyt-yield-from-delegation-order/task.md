## Context

A Python generator is a coroutine-like iterator backed by a hidden state machine.
It has three states — *suspended*, *executing*, and *closed* — and a frame
object that holds the local namespace across suspensions.

Calling `next(g)` transitions the generator from *suspended* to *executing*,
runs the body until the next `yield` expression, emits the yielded value, and
suspends again.  When the generator body falls off the end or hits an explicit
`return v`, Python raises

$$\texttt{StopIteration}(v)$$

where $v$ is the *return value* (available as `e.value` on the exception
object).  For generators without an explicit `return`, the value is `None`.

The `yield from sub` expression (*delegating yield*) creates a direct channel
between the outer generator and the sub-iterator `sub`.  Its semantics are
approximately:

$$\texttt{yield from sub} \;\;\longleftrightarrow\;\;
\texttt{for } v \texttt{ in sub: yield } v$$

but with two crucial additions:

1. **Return-value capture.**  When `sub` raises `StopIteration(v)`, the
   expression `yield from sub` evaluates to $v$ instead of propagating the
   exception:

   $$\texttt{result} = \texttt{yield from sub} \implies
   \texttt{result} = \texttt{StopIteration.value when sub finishes.}$$

2. **Transparent forwarding.**  Calls to `send()`, `throw()`, and `close()`
   on the outer generator are forwarded through the delegation chain to `sub`.

When `yield from` is nested — i.e., a sub-iterator itself delegates to another
sub-iterator — the value stream is *flattened*: every intermediate `yield` from
the innermost generator surfaces in the same order, and each level captures the
`StopIteration.value` of its direct child.

The result of driving a generator $g$ to completion is a pair

$$(\,[v_1, v_2, \ldots, v_n],\; r\,)$$

where $v_1 \dots v_n$ are the yielded values and $r$ is the top-level
`StopIteration.value` (or `None`).

## Task

Implement `collect_yield_from(gen)`:

```python
def collect_yield_from(gen):
    """Drive gen to completion.

    Return (values, return_value) where:
      values:       list of all yielded values in order
      return_value: the StopIteration value from gen (None if unset)
    """
```

The argument `gen` is an already-created generator object.  Drive it to
exhaustion by calling `next()` repeatedly, collect every yielded value into a
list, and — when `StopIteration` is raised — extract the `.value` attribute.

Do **not** use `yield from`, `for ... in gen`, or `list(gen)` inside your
implementation.  Use an explicit `next()` loop and catch `StopIteration`
directly, so you demonstrate knowledge of the exception-based return-value
protocol.

## Example

```python
def _demo():
    def inner():
        yield 10
        return 'ret'
    result = yield from inner()
    yield result

vals, rv = collect_yield_from(_demo())
print(vals)   # [10, 'ret']
print(rv)     # None  (outer generator has no explicit return)
```

Here `inner()` yields `10`, then returns `'ret'`.  The `yield from inner()`
expression captures `'ret'` as `result`, which the outer generator then yields.
The outer generator itself finishes without a `return`, so the final
`StopIteration.value` is `None`.

## What the gate checks

The gate runs five test generators through both the learner's implementation and
a CPython reference oracle that catches `StopIteration` and reads `.value`.
The tests cover:

| # | Pattern |
|---|---------|
| 1 | Plain generator with no `yield from` |
| 2 | Single `yield from` without a return value |
| 3 | Single `yield from` with a return value captured and re-yielded |
| 4 | Nested `yield from` (two delegation levels) with returns at each level |
| 5 | Chained sequential `yield from` calls, each with a return |

The gate metric is **exact_match**: the learner's `(values, return_value)` tuple
must equal the oracle's for every test case.  Any deviation — missed return
values, wrong ordering, or swallowed exceptions — fails the gate.
