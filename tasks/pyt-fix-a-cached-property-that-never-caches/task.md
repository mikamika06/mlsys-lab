## Context

Python attribute access is controlled by descriptors. A descriptor implementing only
`__get__` is a non-data descriptor, which means an instance attribute with the same
name can shadow it. This behavior is useful for cached properties:

$$
\text{first access} \rightarrow \text{compute value} \rightarrow
\text{store in instance dictionary} \rightarrow \text{reuse value}.
$$

A descriptor that also defines `__set__` becomes a data descriptor. Data descriptors
take priority over entries in the instance dictionary, so the cached value is never
used and the computation runs again on every access.

The bug in this task is that a custom `cached_property` is accidentally implemented
as a data descriptor. The fix is to make the descriptor allow the instance dictionary
to hold the cached result.

## Task

Implement `cached_property` as a decorator.

The decorator must return a descriptor object with this behavior:

```python
class Example:
    calls = 0

    @cached_property
    def value(self):
        Example.calls += 1
        return 42
```

For an instance `x = Example()`:

- The first `x.value` call must execute the function and return its result.
- Later `x.value` calls must return the cached value without executing the function again.
- The cached value must be stored on the instance so normal descriptor lookup can use it.
- The descriptor must not prevent instance attributes from shadowing the property.

The decorated function name should be preserved as the attribute name used for caching.

## Example

```python
class User:
    @cached_property
    def score(self):
        print("computing")
        return 100

u = User()
print(u.score)
print(u.score)
```

The output is:

```
computing
100
100
```

The message is printed only once because the second lookup reads the cached instance
attribute.

## What the gate checks

The gate creates classes using the submitted `cached_property` and compares the
number of computations against a reference implementation of a non-data descriptor.

It also records Python line events while repeatedly reading a cached attribute. A
broken data descriptor recomputes the value on every read and produces many more
line events. The submitted implementation must keep the line count below the gate
limit while preserving the correct cached behavior.
