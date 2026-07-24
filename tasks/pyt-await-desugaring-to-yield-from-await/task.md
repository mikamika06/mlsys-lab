## Context

An `await` expression works by obtaining an iterator from the awaited object's
`__await__()` method. The resulting iterator is then driven by the coroutine
machinery. For a custom awaitable object $x$, the core relationship is:

$$
\operatorname{await}\ x \quad \sim \quad \operatorname{yield\ from}\ x.__await__().
$$

The similarity is not that `await` literally expands into source code. Instead,
both mechanisms consume the iterator returned by `__await__()` and use the value
returned when that iterator finishes.

A generator delegation expression can observe two things:

$$
\text{yielded values} = [y_0, y_1, \dots, y_{k-1}]
$$

and the final value returned by the iterator:

$$
\text{return value} = r .
$$

This task uses a custom awaitable whose `__await__()` method is deterministic.
The goal is to expose that the values driven by `await` match the values driven
by delegating with `yield from`.

## Task

Implement `await_desugar_trace(values)`:

```python
def await_desugar_trace(values):
    ...
```

`values` is a list of strings. Return a tuple with four items:

1. A list of values yielded while a coroutine executes `await` on the custom
   awaitable.
2. The final value returned by that `await` expression.
3. A list of values yielded while a generator uses `yield from` on the
   awaitable's `__await__()` iterator.
4. The final value returned by that `yield from`.

The implementation must demonstrate the equivalence between driving
`await expr` and driving `yield from expr.__await__()`.

## Example

```python
trace = await_desugar_trace(["a", "b"])

# (
#   ["a", "b"],
#   "done",
#   ["a", "b"],
#   "done"
# )
```

## What the gate checks

The gate creates several custom awaitables and computes the expected behavior
using CPython's coroutine execution semantics. It checks that the returned tuple
exactly matches the oracle output.

The gate verifies both the yielded sequence and the final return value, so an
implementation that only iterates `__await__()` or only returns the final value
does not pass.
