## Context

Python attribute lookup treats descriptors with different precedence depending on whether
they are data descriptors or non-data descriptors.

A descriptor is a data descriptor when its class defines `__set__` or `__delete__`.
Methods such as `__get__` alone create non-data descriptors. This distinction affects
whether an instance dictionary entry can override the descriptor.

For a descriptor object $d$, the classification can be written as

$$
\texttt{is\_data}(d) =
(\mathrm{hasattr}(\mathrm{type}(d), "\texttt{\_\_set\_\_}"))
\lor
(\mathrm{hasattr}(\mathrm{type}(d), "\texttt{\_\_delete\_\_}")) .
$$

The check uses actual Python descriptor classes and compares your implementation with
the result computed from Python's own class metadata.

## Task

Implement `classify_descriptors(classes)`:

```python
def classify_descriptors(classes):
    ...
```

`classes` is a list of descriptor classes. Return a list of booleans of the same length.
For each class `C`, return `True` when instances of `C` are data descriptors and `False`
when they are non-data descriptors.

You may inspect Python class attributes. Do not instantiate the descriptors or rely on
descriptor behavior during attribute access.

## Example

```python
class ReadOnly:
    def __get__(self, obj, owner):
        return 1

class ReadWrite:
    def __get__(self, obj, owner):
        return 1
    def __set__(self, obj, value):
        pass

result = classify_descriptors([ReadOnly, ReadWrite])
# [False, True]
```

## What the gate checks

The gate builds 15 real descriptor classes with different combinations of `__get__`,
`__set__`, and `__delete__` methods. The returned boolean list must exactly match an
oracle computed from Python's descriptor protocol rules.

The metric `exact_match` is the fraction of descriptor classifications that match the
oracle. A value of $1.0$ is required.
