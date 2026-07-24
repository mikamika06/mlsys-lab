## Context

A Python property is a data descriptor. A descriptor object controls attribute access by defining methods such as `__get__`, `__set__`, and `__delete__`.

The builtin `property()` stores getter, setter, and deleter functions and changes behavior depending on whether an attribute is accessed through an instance or through a class. For an instance access, the getter receives the instance. For a class access, the descriptor itself is returned.

A descriptor can be modeled as a mapping from an object and an operation to a result:

$$
\mathrm{descriptor}(obj, operation) \rightarrow \mathrm{value}
$$

A property implementation must preserve the same observable behavior as the builtin object for supported operations.

## Task

Implement `property_from_scratch(fget, fset=None, fdel=None, doc=None)`.

The function must return a custom property descriptor that matches the important behavior of Python's builtin `property()`:

- Reading `obj.attr` calls `fget(obj)`.
- Assigning `obj.attr = value` calls `fset(obj, value)`.
- Deleting `del obj.attr` calls `fdel(obj)`.
- Reading `Class.attr` returns the descriptor object instead of calling `fget`.
- Missing getter, setter, or deleter operations should raise `AttributeError`.

The returned object should be a descriptor implemented without using the builtin `property` type.

## Example

```python
def get_x(obj):
    return obj._x

def set_x(obj, value):
    obj._x = value

def del_x(obj):
    del obj._x

x = property_from_scratch(get_x, set_x, del_x)

class Box:
    prop = x

b = Box()
b._x = 3

print(b.prop)
b.prop = 7
print(b.prop)
del b.prop
print(Box.prop is x)
```

The output behavior should match the equivalent class using:

```python
property(get_x, set_x, del_x)
```

## What the gate checks

The gate builds the reference behavior using CPython's builtin `property()` and compares it with the submitted implementation.

It checks instance reads, writes, deletes, class-level descriptor access, and missing accessors. The returned value vector must exactly match the builtin oracle behavior, producing an `exact_match` score of $1.0$.
