## Context

Python classes using `__slots__` can avoid creating an instance dictionary. A normal instance stores attributes in a per-object dictionary, while a slotted instance stores them in fixed descriptors defined by the class layout.

For a class instance, the memory footprint can be compared with

$$
\mathrm{size\_ratio} = \frac{\mathrm{measured\_size}}{\mathrm{target\_size}} .
$$

A subclass can accidentally defeat a slotted base class. If a subclass does not define `__slots__`, CPython gives the subclass its own `__dict__`, which increases the size of every instance.

Consider a base class with fixed slots:

```python
class Base:
    __slots__ = ("value",)

class Broken(Base):
    pass
```

`Broken` instances regain a dictionary even though `Base` was slotted.

## Task

Implement `restore_slots(cls)`.

The function receives a subclass that accidentally reintroduced instance dictionaries and returns a replacement subclass with the same behavior but a slotted instance layout. The returned class must not allocate `__dict__` storage for instances.

Requirements:

- Preserve the original class name and methods.
- Keep the original base classes.
- Add an empty `__slots__` declaration to the replacement class.
- Do not copy `__dict__` or `__weakref__` descriptors from the broken class.

## Example

```python
class Base:
    __slots__ = ("value",)

class Broken(Base):
    def double(self):
        return self.value * 2

Fixed = restore_slots(Broken)

obj = Fixed()
obj.value = 5
assert obj.double() == 10
assert not hasattr(obj, "__dict__")
```

## What the gate checks

The gate creates broken subclasses and compares the returned class instance footprint against a real CPython slotted-class oracle using `sys.getsizeof`.

The reported `size_ratio` is

$$
\frac{\mathrm{size\ of\ returned\ instance}}{\mathrm{size\ of\ oracle\ slotted\ instance}} .
$$

A ratio close to $1$ is required. A subclass that still has a `__dict__` is larger and fails the gate.
