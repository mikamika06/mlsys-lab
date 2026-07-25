## Context

Python objects can store instance attributes in different ways. A normal class instance usually has a separate instance dictionary, while a class using `__slots__` can store declared attributes without an instance `__dict__`.

The memory contribution of a normal instance can be approximated as

$$
S_{\mathrm{dict}} = \operatorname{getsizeof}(x) + \operatorname{getsizeof}(x.\texttt{\_\_dict\_\_}),
$$

because both the object header and the attribute dictionary occupy memory. A slotted instance has no instance dictionary, so the comparable measurement is

$$
S_{\mathrm{slots}} = \operatorname{getsizeof}(y).
$$

The ratio

$$
R = \frac{S_{\mathrm{dict}}}{S_{\mathrm{slots}}}
$$

shows how much larger the measured storage of the dictionary-backed instance is compared with the slotted instance on the current CPython build.

## Task

Implement `dict_vs_slots_size_ratio()`:

```python
def dict_vs_slots_size_ratio() -> float:
    ...
```

Create one normal class and one class using `__slots__`. Each class should define the same small set of instance attributes. Create one instance of each class and return the ratio

$$
\frac{\operatorname{getsizeof}(\text{dict instance}) + \operatorname{getsizeof}(\text{dict instance}.\texttt{\_\_dict\_\_})}
{\operatorname{getsizeof}(\text{slotted instance})}.
$$

Use `sys.getsizeof` for the measurements. The function must work on the CPython version used by the grader without assuming a fixed byte size.

## Example

```python
ratio = dict_vs_slots_size_ratio()
# Example shape only:
# ratio > 1.0
```

The exact value depends on the CPython object layout, so the implementation must measure it instead of returning a constant.

## What the gate checks

The gate builds the same measurement using the real CPython `sys.getsizeof` behavior as an oracle. Your returned ratio is compared against that measured ratio.

Returning a value based on hardcoded object sizes or omitting the instance dictionary contribution will fail because the measured ratio will not match the oracle.
