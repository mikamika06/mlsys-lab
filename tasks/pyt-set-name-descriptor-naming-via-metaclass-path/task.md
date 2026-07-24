## Context

A descriptor object is created once, at class-body execution time, before it
knows which attribute name it will end up bound to:

```python
class Widget:
    width = NamedField()   # NamedField() has no idea it's called "width" yet
```

PEP 487 closes this gap: when a class body finishes executing, `type.__new__`
walks every entry in the new class's namespace and, for any value whose type
defines `__set_name__`, calls `value.__set_name__(owner_class, attr_name)`.
This is the hook a descriptor uses to learn the name it was assigned to,
without the class author having to repeat it (`width = NamedField("width")`).

A metaclass sits at exactly this point in class creation — its `__new__`
receives the finished namespace dict before returning the class object — so
it is a natural place to build a **registry** of the descriptors a class
declares, keyed by the same names `__set_name__` just recorded. This is the
mechanism behind libraries that discover a class's declared fields (ORMs,
serializers, dataclass-like tools) without any explicit registration step
from the user.

## Task

Implement two names:

```python
class NamedField:
    """Descriptor that learns its attribute name via __set_name__ and
    stores each instance's value under a private per-instance attribute
    (e.g. "_" + name)."""
    def __set_name__(self, owner, name): ...
    def __get__(self, obj, objtype=None): ...
    def __set__(self, obj, value): ...

class FieldMeta(type):
    """Metaclass that, after building the class, sets cls._fields to a
    tuple of the names of every NamedField declared directly in the class
    body, in declaration order (non-NamedField attributes are excluded)."""
    def __new__(mcs, name, bases, namespace):
        ...
```

`NamedField` must record the name it is bound to (via `__set_name__`) and use
it to store/retrieve each instance's value from a private attribute, so two
different `NamedField` attributes on the same class do not collide. `get`
on the class itself (`obj is None`) should return the descriptor object.
`FieldMeta.__new__` must call `super().__new__` to actually build the class
(so the normal `__set_name__` dispatch still happens), then scan the
namespace it was given for `NamedField` instances and attach `cls._fields`.

## Example

```python
class Widget(metaclass=FieldMeta):
    width = NamedField()
    height = NamedField()

Widget._fields          # ("width", "height")
w = Widget()
w.width = 10
w.width                  # 10
Widget.__dict__["width"].name   # "width"
```

## What the gate checks

The grader dynamically builds several classes with `metaclass=FieldMeta` and
a mix of `NamedField` and plain attributes, then checks, using
`exact_match`:

- `cls._fields` matches the declared `NamedField` attribute names, in
  declaration order, excluding non-`NamedField` attributes;
- each field descriptor's recorded name (set by `__set_name__`) equals the
  attribute name it was actually assigned to;
- reading and writing fields on two different instances of the same class
  never cross-contaminate (per-instance, not per-class, storage);
- an empty class produces `_fields == ()`, and building one class's field
  list does not disturb another already-built class's `_fields`.
