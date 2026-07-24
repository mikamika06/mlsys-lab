## Context

Python creates every class through a metaclass. When a class inherits from multiple
base classes, the metaclasses of those bases must be compatible. A conflict occurs
when Python cannot find a single metaclass that is a subclass of all required
metaclasses.

If classes require metaclasses $M_1$ and $M_2$, a valid combined metaclass $M$ must
satisfy

$$M \subseteq M_1 \quad \text{and} \quad M \subseteq M_2,$$

where the subset notation means "is a subclass of" in the metaclass hierarchy.

A common fix is to introduce a new metaclass that inherits from both conflicting
metaclasses. The new metaclass becomes the creator for the final class while the
original metaclass behavior remains available through inheritance.

## Task

Implement `resolve_mro_names()`:

```python
def resolve_mro_names() -> list[str]:
    ...
```

The function must create a multiple-inheritance class from two bases that originally
have incompatible metaclasses. Resolve the conflict by introducing a compatible
combined metaclass.

Return the names in the method resolution order (MRO) of the resulting class as a
list of strings. The returned list should come from the final constructed class's
`__mro__` attribute.

## Example

A successful implementation constructs a class hierarchy equivalent to:

```python
names = resolve_mro_names()
# ["Combined", "Left", "Right", "object"]
```

The exact returned list is determined by Python's MRO calculation.

## What the gate checks

The gate builds the same conflicting metaclass situation using the CPython class
creation machinery and computes the expected MRO after applying a valid combined
metaclass. Your function output is compared against that computed result.

The check requires an exact match of the MRO name sequence. Implementations that
still raise a metaclass conflict or return a manually typed list will fail.
