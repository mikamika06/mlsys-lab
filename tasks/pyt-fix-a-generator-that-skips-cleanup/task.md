## Context

A Python generator that never runs to completion doesn't just vanish
silently. When the last reference to it is dropped, CPython's refcounting
deallocator calls `close()` on it automatically; the same thing happens if
the caller calls `.close()` explicitly. Either way, `close()` resumes the
suspended generator by throwing `GeneratorExit` in at the point of the last
`yield`.

If cleanup code sits in a plain statement *after* the loop that yields, that
statement is only ever reached when the generator is iterated to
exhaustion — the normal control-flow path. It is **never** reached if the
generator is abandoned or closed early, because `GeneratorExit` unwinds the
frame from the suspended `yield` straight out, skipping every statement
that would otherwise have run later. The fix is to put the cleanup in a
`try/finally` around the yielding code: `finally` runs on *every* exit path
out of the `try` block — normal completion, an explicit `close()`, or
garbage collection — because Python implements `finally` at the bytecode
level as an unconditional cleanup handler, not as "the next line if nothing
went wrong."

## Task

Implement `make_managed_gen(events, n)`:

```python
def make_managed_gen(events: list, n: int):
    ...
```

Return a generator (or any iterator implementing the same protocol) that,
when driven:

- Appends the string `"acquire"` to `events` the first time it is resumed
  (i.e. on the first `next()`/`send()`, not at creation time — a generator
  body doesn't run until first resumed).
- Yields the integers `0, 1, ..., n-1` in order, one per resume.
- Appends the string `"release"` to `events` **exactly once**, no matter
  which of these ends the generator: running off the end after yielding all
  `n` values, an explicit `.close()` call, or the generator being garbage
  collected after only some (or none) of the values were consumed.

`events` must never end up missing `"release"`, and must never contain it
more than once.

## Example

```python
events = []
g = make_managed_gen(events, 5)
next(g)
next(g)
g.close()
# events == ["acquire", "release"]  -- released even though only 2/5 consumed
```

## What the gate checks

The gate is entirely self-checking against real CPython generator/GC
semantics — no external oracle is needed because `close()`,
`GeneratorExit`, and reference-counted collection are the language's own
documented, deterministic behavior. It drives your `make_managed_gen` through
three scenarios and requires all of them to hold:

1. **Full consumption** — exhaust every value; `events` must end up exactly
   `["acquire", "release"]`, with the release recorded only after the
   `"acquire"`.
2. **Early explicit close** — consume 2 of `n=5` values, then call
   `.close()`; `"release"` must appear in `events` even though the
   generator never reached the end of its values.
3. **Abandonment** — consume 1 value, then drop every reference to the
   generator (`del g` followed by `gc.collect()` as a safety net for cyclic
   references); `"release"` must still appear.

The gate metric is `1.0` only if all three scenarios produce exactly the
expected event sequence, else `0.0`. A generator whose cleanup line sits
after the yielding loop instead of in a `finally` passes scenario 1 but
fails scenarios 2 and 3, because `GeneratorExit` unwinds past that line
without ever executing it.
