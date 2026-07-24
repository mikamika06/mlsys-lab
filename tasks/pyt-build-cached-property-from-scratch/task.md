## Context

A **descriptor** is an object whose class defines `__get__`, `__set__`, or
`__delete__`. A *non-data* descriptor defines only `__get__`; a *data*
descriptor also defines `__set__` (or `__delete__`). When Python resolves
`instance.name`, the lookup order is:

1. a **data** descriptor found on the class,
2. an entry in `instance.__dict__`,
3. a **non-data** descriptor found on the class.

So a non-data descriptor whose `__get__` writes its computed value straight
into `instance.__dict__` is invoked only **once**: from then on, step 2
finds the plain dict entry first and `__get__` is never reached again. This
is exactly how `functools.cached_property` works.

To learn the attribute's own name without the caller repeating it (e.g.
`self.func.__name__`, which breaks under `staticmethod`-style wrapping and
subclassing edge cases), Python calls `__set_name__(owner, name)` on every
descriptor found in a class body, once, right after the class is created —
passing the class and the attribute name it was assigned under.

## Task

Implement the class `cached_property`:

```python
class cached_property:
    def __init__(self, func):
        ...
    def __set_name__(self, owner, name):
        ...
    def __get__(self, instance, owner=None):
        ...
```

* `__init__(self, func)` stores the wrapped zero-argument method `func`.
* `__set_name__(self, owner, name)` is called automatically by Python when
  the class body finishes executing; record `name` (the attribute name the
  descriptor was assigned to) on `self`.
* `__get__(self, instance, owner=None)`:
  1. If `instance is None` (attribute accessed on the class itself),
     return the descriptor object itself.
  2. Otherwise call `self.func(instance)` to compute the value, store it in
     `instance.__dict__` under the recorded name, and return it.

Do **not** define `__set__` — the class must remain a non-data descriptor,
so that instance-dict entries (written either by your `__get__` or by
anyone else) always take priority over it on later lookups.

## Example

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @cached_property
    def norm(self):
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3, 4)
print(p.norm)       # 5.0  (computes, then caches)
print(p.norm)       # 5.0  (found directly in p.__dict__, func not called)
print(p.__dict__)   # {'x': 3, 'y': 4, 'norm': 5.0}
```

## What the gate checks

Two gates run against a dynamically built test class whose methods each
append `id(self)` to a plain Python list every time they actually execute
— a real, directly observable invocation count, not a hardcoded number.

* **op_count** (`== 0`): each property is accessed several times per
  instance, across two separate instances. `op_count` is the number of
  *extra* (duplicate) computations beyond exactly one per (instance,
  property) pair — nonzero if the value is recomputed on every access.
* **exact_match** (`== 1.0`): checks that returned values are correct, that
  two instances cache independently (no shared/class-level cache), that
  the cached value really lives in `instance.__dict__`, that overwriting
  `instance.__dict__[name]` directly is honored on the next access (the
  signature of a genuine non-data descriptor — a data descriptor with a
  stray `__set__` would ignore it), and that class-level access returns a
  descriptor-like object instead of raising or computing.
