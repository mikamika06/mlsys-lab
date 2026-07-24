## Context

Python classes have a method resolution order (MRO) that defines the order in
which attribute lookup searches classes. For a class `cls`, CPython exposes this
ordering through `cls.__mro__`.

The MRO is produced by C3 linearization. For a class hierarchy with parents
$B_1, B_2, \dots, B_k$, the algorithm computes a consistent sequence that keeps
local precedence order while preserving monotonicity.

The value of `cls.__mro__` is a tuple of class objects:

$$
\mathrm{cls.__mro__} = (C_0, C_1, \dots, C_n),
$$

where $C_0$ is `cls` itself and later entries are its base classes and
ancestors. Reading the class names from this tuple gives a compact way to inspect
the inheritance order.

## Task

Implement `mro_names(cls)`:

```python
def mro_names(cls):
    ...
```

The function receives a Python class object and returns a tuple containing the
string names of the classes in `cls.__mro__`, in the same order.

Do not compute the inheritance order manually. Use the class metadata exposed by
Python.

## Example

```python
class Root:
    pass

class Left(Root):
    pass

class Right(Root):
    pass

class Child(Left, Right):
    pass

mro_names(Child)
# ("Child", "Left", "Right", "Root", "object")
```

## What the gate checks

The gate creates fixture class hierarchies and compares the returned tuple with
the tuple of names read from the real CPython `cls.__mro__` attribute. The result
must match exactly.
