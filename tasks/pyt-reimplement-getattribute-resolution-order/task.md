## Context

`instance.name` doesn't just look in one place. CPython's default
`object.__getattribute__` checks, in order:

$$
\text{data descriptor in MRO} \;>\; \text{instance } \verb|__dict__| \;>\; \text{non-data descriptor in MRO} \;>\; \text{plain class attribute in MRO} \;>\; \verb|__getattr__|
$$

A **descriptor** is any object whose *type* defines `__get__`. It is a
**data descriptor** if its type *also* defines `__set__` or
`__delete__`; otherwise it's a **non-data descriptor**. This
distinction is exactly why `@property` (a data descriptor) always wins
over `self.__dict__`, while a plain method (a non-data descriptor,
since a function's type only has `__get__`) can be shadowed by
assigning an instance attribute of the same name.

Walking `type(obj).__mro__` (rather than, say, `type(obj).__bases__`
recursively) matters too: for diamond inheritance, the MRO is what
determines which ancestor's attribute is found first, and it is not
always simple left-to-right depth-first order.

## Task

Implement `resolve`:

```python
def resolve(obj, name: str):
    ...
```

* `obj` — any instance.
* `name` — attribute name to resolve.
* Returns the value CPython would give you for `obj.name`, or raises
  `AttributeError(name)` if nothing resolves it.

Implement the precedence chain above by hand: walk `type(obj).__mro__`
looking for `name` in each `klass.__dict__`; check the descriptor
protocol via `hasattr(type(value), '__get__')` /
`hasattr(type(value), '__set__' or '__delete__')` on whatever you find;
consult `obj.__dict__` directly; and search the MRO again for a
`__getattr__` to use as the final fallback.

**Do not** call `getattr`, `super()`, `obj.__getattribute__`, or
`object.__getattribute__` to resolve `name` — that would just delegate
the problem back to the exact machinery you're reimplementing. (Using
`hasattr`/`type(...)` to inspect the *descriptor protocol* on some
other name, like `__get__`, is fine and unavoidable.)

## Example

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        return "from descriptor" if obj is not None else self
    def __set__(self, obj, value):
        pass   # having __set__ makes this a DATA descriptor

class C:
    x = Descriptor()
    def __init__(self):
        self.__dict__['x'] = "from instance dict"   # would normally lose

c = C()
resolve(c, 'x')   # -> "from descriptor"   (data descriptor beats instance dict)
```

If `Descriptor` did *not* define `__set__`, it would be a non-data
descriptor, and `resolve(c, 'x')` would instead return
`"from instance dict"`.

## What the gate checks

* **exact_match** — the grader builds real classes exercising every
  tier of the precedence chain (data descriptor vs. instance dict,
  non-data descriptor shadowed by instance dict, plain class attribute,
  diamond-inheritance MRO order, `__getattr__` fallback, and a genuine
  miss that must raise `AttributeError`), and compares your `resolve`
  output against **real `getattr(obj, name)`** on those same real
  objects — the actual CPython interpreter is the oracle. Must be
  `1.0` (10/10 fixtures, including matching the `AttributeError` case).
* **no_shortcut** — inspects your submitted function's bytecode
  (`__code__.co_names`, including any nested/closure code objects) to
  confirm it never references `getattr`, `super`, or
  `__getattribute__`. Must be `1.0`.
