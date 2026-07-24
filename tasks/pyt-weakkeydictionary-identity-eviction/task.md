## Context

A plain `dict` holds a strong reference to every key it contains, so using
objects as cache keys keeps them alive for as long as the cache exists — even
if nothing else in the program still needs them. This is a common source of
accidental memory leaks in caches keyed by application objects.

`weakref.WeakKeyDictionary` solves this by holding only a *weak* reference to
each key. A weak reference does not count toward an object's reference
count, so when the last strong reference to a key object goes away, CPython's
reference-counting collector finalizes it immediately (assuming no reference
cycle is involved), and the `WeakKeyDictionary` automatically drops the
now-dangling entry — no explicit cleanup call required. The cache's key set
tracks the identity and lifetime of the actual key objects: two entries are
different if they were put in under two different objects, even if those
objects would compare `==`, and an entry disappears exactly when its key
object dies.

## Task

Implement `IdentityCache`:

```python
class IdentityCache:
    """Identity-keyed cache; an entry auto-evicts when its key object dies."""
    def __init__(self):
        ...

    def put(self, key, value):
        ...

    def get(self, key, default=None):
        ...

    def __len__(self):
        ...
```

Back the cache with `weakref.WeakKeyDictionary` so entries evict themselves
automatically the moment their key object is garbage collected — do not
implement manual bookkeeping (e.g. polling `sys.getrefcount`) to detect
death. `put` stores `value` under `key`; `get` returns the stored value for
`key`, or `default` if `key` is not present (either never inserted or
already evicted); `len(cache)` returns the number of currently live entries.

## Example

```python
class Obj:
    pass

cache = IdentityCache()
a = Obj()
cache.put(a, "hello")
len(cache)          # 1
del a                # last strong reference to the key object is gone
import gc; gc.collect()
len(cache)          # 0 — the entry evicted itself
```

## What the gate checks

The grader creates several plain key objects, puts each into your cache with
a distinct value, and confirms `len(cache)` matches the number inserted.
It then drops the only strong reference to a subset of the key objects
(setting the local variables to `None`), runs `gc.collect()`, and checks that
`len(cache)` dropped to exactly the number of survivors, that `get` still
returns the correct value for every surviving key, and that dropping the
remaining keys brings the cache down to `len(cache) == 0`. All of this must
match exactly (`exact_match`); a cache that stores keys with a plain `dict`
(so nothing is ever evicted) or one that evicts too eagerly both fail.
