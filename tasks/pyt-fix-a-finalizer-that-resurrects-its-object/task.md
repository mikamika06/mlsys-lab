## Context

CPython collects an object the instant its reference count hits zero — but
`__del__` runs *before* the memory is actually freed, and `__del__` can
create a brand-new strong reference to `self` (e.g. by appending it to a
module-level list). When that happens the refcount goes back above zero and
the object is **resurrected**: it survives, still fully alive, even though
every caller believes it was destroyed. This is a classic footgun with
`object.__del__`.

The broken `Resource` in `starter.py` does exactly this:

```python
_GRAVEYARD = []

class Resource:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def __del__(self):
        self.events.append(("finalized", self.name))
        _GRAVEYARD.append(self)          # BUG: resurrects itself
```

`weakref.finalize(obj, callback, *args)` is the fix: it registers `callback`
against a *weak* reference to `obj`, guaranteed to fire **exactly once**, and
the callback must only close over plain data (never `obj` itself) — so there
is nothing left for it to resurrect.

## Task

Fix `Resource` (and, if needed, `resurrection_safe_lifecycle`) so that
finalization never resurrects the object. Implement:

```python
def resurrection_safe_lifecycle() -> list[tuple]:
    ...
```

which must run this exact scenario for `name` in `("A", "B")`:

1. Create `r = Resource(name, events)`.
2. Take `ref = weakref.ref(r)`.
3. Put `r` into a one-element list `holder = [r]`, then `del r` (only
   `holder` keeps it alive).
4. `holder.clear()` — the refcount hits zero right here, synchronously (no
   reference cycle is involved anywhere, so plain CPython refcounting is
   enough; no `gc.collect()` needed).
5. Append `("alive_after_drop", name, ref() is not None)` to `events`.

Return `events`. On a correctly finalized (non-resurrecting) `Resource`,
`ref()` must be `None` at step 5 — the object is truly gone.

## Example

```python
resurrection_safe_lifecycle()
# -> [('finalized', 'A'), ('alive_after_drop', 'A', False),
#     ('finalized', 'B'), ('alive_after_drop', 'B', False)]
```

The broken version instead produces `('alive_after_drop', 'A', True)` (and
the same for `'B'`): the object's own `__del__` re-attached it to a global
list, so the "destroyed" resource is actually still alive.

## What the gate checks

`exact_match` — the grader independently computes the same event sequence
with a correct `weakref.finalize`-based implementation and compares it,
element for element, to your `resurrection_safe_lifecycle()`'s return value.
Any difference — a missing/duplicated `"finalized"` event, or an
`alive_after_drop` flag of `True` where it should be `False` — fails the gate.
