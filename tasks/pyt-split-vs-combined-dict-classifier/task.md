## Context

CPython dictionaries have two internal table layouts:

- **Combined**: a single table holds keys, hashes, and values together.
  Every plain `dict` literal, `dict(...)` call, and dict comprehension
  produces a combined dict.
- **Split**: the values live in a separate `ma_values` array, while the
  keys/hashes live in a `ma_keys` table that can be *shared* between
  multiple dicts — most commonly the `__dict__` of several instances of
  the same class, so that instances with the same attribute names don't
  each pay for their own copy of the key table.

CPython's own source treats this structurally: a dict is split exactly
when its `ma_values` pointer is non-`NULL`; it is combined when
`ma_values` is `NULL` (values then live inline inside `ma_keys`). This is
literally the condition CPython uses internally (`_PyDict_HasSplitTable`
in `Objects/dictobject.c`).

The trigger conditions are less obvious than intuition suggests. For
instance:

$$
\text{del obj.attr} \;\not\Rightarrow\; \text{combined}
$$

Deleting an attribute does **not** force a combine — CPython can leave a
hole in the shared-keyed values array. But other operations do force a
combine, e.g. inserting a non-string key directly into `obj.__dict__`, or
calling `.update(...)` on an instance dict. The only way to know for sure
is to ask the live object, not to guess from a rule of thumb.

## Task

Implement `is_split_dict`:

```python
def is_split_dict(d: dict) -> bool:
    ...
```

Given any `dict` object `d`, return `True` if CPython currently
represents it as a **split-table** dict, and `False` if it is a
**combined** dict.

You must determine this from the live object's actual internal
representation (e.g. by reading the `PyDictObject` header with `ctypes`)
rather than by guessing from surface-level rules such as "was this a
dict literal" or "did anyone call `del`" — several of the test cases are
specifically designed to break naive rules like that.

## Example

```python
class Point:
    pass

p = Point()
p.x, p.y = 1, 2

is_split_dict(p.__dict__)        # -> True  (shares keys with the class)
is_split_dict({"x": 1, "y": 2})  # -> False (plain literal, combined)

p.__dict__.update({"z": 3})
is_split_dict(p.__dict__)        # -> False (.update() forces a combine)
```

## What the gate checks

The gate builds a dozen dicts through different real code paths (fresh
instance attributes, a second instance of the same class, plain
literals, `dict(...)`, a dict comprehension, an instance dict after
`del`, a `dict()` copy of an instance dict, an instance dict with a
non-string key inserted directly, an empty not-yet-populated instance
dict, `vars(instance)`, an instance dict built via `.update()`, and a
`dict` subclass instance).

For each one it computes the ground-truth split/combined label by
reading the actual `PyDictObject.ma_values` pointer of that exact live
object at grading time — never a hardcoded or remembered answer — and
compares it against your function's return value. All cases must match
exactly (`exact_match == 1.0`).
