## Context

Python iterators implement the iterator protocol through two methods:

- `__iter__()` returns the iterator object itself.
- `__next__()` returns the next produced value and raises `StopIteration` when no values remain.

An iterator represents a stateful process. If an iterator has produced values $x_0, x_1, \dots, x_{n-1}$, then calling `next()` repeatedly should return those values in order and then signal completion.

The built-in `range` object is an example of an iterable sequence generator. A custom iterator often stores the current position and updates it after each successful `__next__()` call.

## Task

Implement the class `CountdownIterator`:

```python
class CountdownIterator:
    def __init__(self, start: int):
        ...

    def __iter__(self):
        ...

    def __next__(self):
        ...
```

The iterator must produce the sequence:

$$start, start-1, start-2, \dots, 1$$

For example, `CountdownIterator(4)` must produce `4, 3, 2, 1` and then raise `StopIteration` forever after the sequence is exhausted.

`__iter__()` must return the same iterator object so that the class follows the Python iterator protocol.

## Example

```python
it = CountdownIterator(3)

list(it)
# [3, 2, 1]

next(it)
# raises StopIteration
```

## What the gate checks

The gate compares the produced sequence against a reference sequence generated from Python's built-in iterator behavior. It also checks that the returned object from `__iter__()` is the iterator itself and that exhaustion raises `StopIteration`.

A class that returns the correct values but is not a valid iterator protocol implementation will fail.
