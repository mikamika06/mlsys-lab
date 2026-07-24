## Context

Python objects have identity and value as separate concepts. The built-in `id()` function returns an integer that identifies an object during its lifetime.

Two objects can contain equal values while still being different objects:

$$
[1,2] = [1,2]
$$

as values, but if they are separately created objects then:

$$
\mathrm{id}(a) \neq \mathrm{id}(b).
$$

Identity-based algorithms use object identity instead of equality comparisons. This matters when tracking shared objects, caches, graph nodes, or references where two equal-looking values should remain separate.

## Task

Implement `identity_partition(objects)`:

```python
def identity_partition(objects):
    ...
```

The function receives a list of arbitrary Python objects. Return a tuple of tuples, where each inner tuple contains the indices of objects that have the same identity.

Objects in the same group must satisfy:

$$
\mathrm{id}(objects_i) = \mathrm{id}(objects_j).
$$

Groups must be ordered by their smallest index. Indices inside each group must be increasing.

Do not group objects using `==`, hashing, or value-based comparisons. The result must depend only on `id()`.

## Example

```python
a = [1, 2]
b = [1, 2]
items = [a, b, a]

identity_partition(items)
# ((0, 2), (1,))
```

The two lists containing `[1, 2]` are equal by value, but only the references to `a` belong to the same identity group.

## What the gate checks

The gate builds lists containing shared references and equal-but-distinct objects. It computes the expected partition using the CPython identity primitive `id()` and compares the returned tuple structure exactly.

A solution that groups by equality or value will fail because objects with the same contents but different identities must remain in different partitions.
