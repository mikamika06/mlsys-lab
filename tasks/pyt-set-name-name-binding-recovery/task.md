## Context

A descriptor object doesn't normally know what name it was assigned to in
a class body:

```python
class Point:
    x = SomeDescriptor()
    y = SomeDescriptor()
```

Inside `SomeDescriptor.__init__`, there's no way to know the future — the
name `"x"` or `"y"` doesn't exist yet when `SomeDescriptor()` runs. CPython
solves this with a dedicated hook: right after a class body finishes
executing, `type.__new__` scans the new class's namespace and calls

$$
\texttt{descriptor.\_\_set\_name\_\_(owner\_class,\ attribute\_name)}
$$

once for **every** attribute in the class whose value defines a
`__set_name__` method — passing the class being created and the exact name
that value was bound to.

The gotcha: this fires once **per binding**, not once per descriptor
instance. If the very same descriptor object is assigned to two different
class attributes, `__set_name__` is called twice on it (in class-body
order), and the second call's name simply overwrites whatever the first
call stored.

## Task

Implement both:

```python
class NamedField:
    """A descriptor that discovers its own attribute name automatically via
    __set_name__. Values are stored per-instance under a private
    ("_" + name) key."""

    def __set_name__(self, owner, name): ...
    def __get__(self, obj, objtype=None): ...
    def __set__(self, obj, value): ...


def recovered_names(cls):
    """Return {class_attribute_name: name_captured_by___set_name__} for
    every NamedField descriptor defined directly on `cls`."""
    ...
```

* `NamedField.__set_name__` must record the given `name` (e.g. as
  `self.name`) and a private storage key (e.g. `self.private_name = "_" +
  name`).
* `__get__`/`__set__` must round-trip a value through `self.private_name`
  on the instance (returning `None` if unset, and never leaving a same-named
  public key in the instance's `__dict__`).
* `recovered_names(cls)` inspects `vars(cls)` and returns a dict mapping
  every attribute name that holds a `NamedField` to that descriptor's
  recovered `.name` — skipping any class attribute that isn't a
  `NamedField` (constants, methods, dunders, ...).

## Example

```python
class Point:
    x = NamedField()
    y = NamedField()

recovered_names(Point)   # -> {"x": "x", "y": "y"}

shared = NamedField()
class Weird:
    a = shared
    b = shared

recovered_names(Weird)   # -> {"a": "b", "b": "b"}
# __set_name__ fired twice on the SAME object — once for "a", once for
# "b" — so only the last call's name survives, for both attributes.
```

## What the gate checks

The grader builds three classes using **your** `NamedField` and calls
**your** `recovered_names` on each:

1. `Point` with two independent `NamedField()` instances — expects
   `{"x": "x", "y": "y"}`.
2. `Weird`, where the exact same `NamedField()` instance is bound to two
   attributes — expects `{"a": "b", "b": "b"}` (last binding wins for a
   shared instance, matching real `__set_name__` semantics).
3. `Mixed`, mixing a `NamedField` with a plain class constant and a method
   — expects only the `NamedField` entry, `{"p": "p"}`.

It also constructs a `Point` instance, assigns `.x` and `.y`, and checks
they round-trip correctly through private storage (no `"x"` key leaking
into the instance's `__dict__`). **exact_match** is `1.0` only if all four
checks pass exactly; any mismatch or exception makes it `0.0`.
