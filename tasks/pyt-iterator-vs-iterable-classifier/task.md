## Context

Python separates the idea of an iterable from the idea of an iterator.

An iterable is an object that can produce an iterator through `iter(x)`. An
iterator is an object that follows the iterator protocol and returns itself from
`iter(x)`:

$$
\operatorname{is\_iterator}(x) = ( \operatorname{iter}(x) \text{ is } x ).
$$

Many containers are iterable but not iterators. For example, a list can create a
new iterator each time, while a list iterator is the iterator object that
maintains traversal state.

## Task

Implement `classify_iterators(objects)`:

```python
def classify_iterators(objects):
    ...
```

The function receives a list of Python objects and returns a list of booleans of
the same length. Each output value must be `True` exactly when calling
`iter()` on the corresponding object returns the same object identity.

Objects that cannot be passed to `iter()` should be classified as `False`.

Do not check concrete types such as `list_iterator` or `generator`. Use the
iterator protocol behavior itself.

## Example

```python
items = [
    [1, 2, 3],
    iter([1, 2, 3]),
    (x for x in range(3)),
]

print(classify_iterators(items))
# [False, True, True]
```

## What the gate checks

The gate creates a collection of different Python objects and computes the
expected labels by asking the real CPython iterator protocol whether
`iter(obj) is obj`.

The returned list from `classify_iterators` must exactly match this oracle
output for every object in the test collection.
