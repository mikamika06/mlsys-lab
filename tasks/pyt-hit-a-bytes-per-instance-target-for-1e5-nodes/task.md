## Context

Python instances usually have an instance dictionary that stores attributes dynamically. This is flexible, but every object pays the memory cost of that dictionary.

For many small objects, a fixed layout can reduce memory usage. The `__slots__` feature removes the per-instance dictionary and stores only the declared attributes.

If $N$ objects are created, the average memory footprint is

$$
\mathrm{bytes\_per\_instance} = \frac{\mathrm{total\_bytes}}{N}.
$$

This task measures a graph node layout over $100000$ instances. The target is the smallest layout that still stores the required fields.

## Task

Implement:

```python
class Node:
    ...
```

The constructor must accept:

```python
Node(value, left, right)
```

and store the values in attributes named:

```python
value
left
right
```

Implement:

```python
def node_size_ratio(n: int = 100000) -> float:
    ...
```

It must measure the memory footprint of your own `Node` implementation and divide it by the footprint of a reference implementation that uses a fixed layout.

The measurement must use `sys.getsizeof` and include the size of an instance dictionary when one exists.

## Example

```python
node = Node(10, None, None)

assert node.value == 10
assert node.left is None
assert node.right is None

ratio = node_size_ratio(100000)
```

A memory-efficient implementation should achieve a ratio close to $1.0$.

## What the gate checks

The gate creates a real CPython oracle class with:

```python
__slots__ = ("value", "left", "right")
```

and measures both the oracle and the submitted `Node`.

The checked value is

$$
\mathrm{size\_ratio}
=
\frac{\mathrm{candidate\ bytes\ per\ instance}}
{\mathrm{oracle\ bytes\ per\ instance}}.
$$

The oracle is computed at runtime using CPython's actual object sizes, so the test does not depend on hardcoded byte counts.

The gate passes only when

$$
\mathrm{size\_ratio} \le 1.0.
$$

A class that keeps a normal instance dictionary will exceed the target.
