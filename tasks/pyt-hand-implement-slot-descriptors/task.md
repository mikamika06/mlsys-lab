## Context

Python instance attributes are often stored in an instance dictionary. A class
using `__slots__` changes this layout by using fixed storage locations accessed
through member descriptors.

A descriptor controls attribute access through methods such as `__get__`,
`__set__`, and `__delete__`. A simplified slot implementation can map each
attribute to one position in a backing array:

$$
s = [s_0, s_1, \dots, s_{k-1}]
$$

where descriptor $i$ reads and writes $s_i$. If the stored value is removed,
reading that slot should raise `AttributeError`.

## Task

Implement `hand_slots_roundtrip()`.

The function must manually implement slot-like behavior without using the
`__slots__` class declaration. Create descriptors that manage fixed positions
in a private backing store.

The function must:

1. Create an object with managed attributes `x`, `y`, and `z`.
2. Assign values `10`, `20`, and `30` through descriptor access.
3. Read the values back.
4. Delete attribute `y` and verify that reading it raises `AttributeError`.
5. Make `instance.__dict__` access raise `AttributeError`.
6. Return:

```python
(values, no_dict)
```

where `values` is the list of values read before deletion and `no_dict` is a
boolean that is true only when dictionary access is blocked and deletion worked.

Do not use `__slots__`.

## Example

```python
result = hand_slots_roundtrip()

# result:
# ([10, 20, 30], True)
```

## What the gate checks

The gate creates an independent descriptor-based CPython oracle and compares the
returned value exactly.

The `exact_match` metric must equal $1.0$. The check verifies that descriptor
writes and reads work, deletion removes an attribute, and the instance dictionary
interface is unavailable.
