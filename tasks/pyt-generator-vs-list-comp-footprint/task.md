## Context

A list comprehension eagerly creates every element and stores the resulting list in memory. A generator expression creates a generator object that stores iteration state and produces values lazily.

For a sequence of length $N$, the memory footprint of the container objects is different:

$$
\text{footprint ratio} = \frac{\operatorname{sizeof}(\text{list comprehension result})}{\operatorname{sizeof}(\text{generator expression})}.
$$

The generator does not contain all $N$ computed values. Instead, it keeps enough state to resume execution when `next()` is called.

Python exposes object sizes through `sys.getsizeof`, which reports the memory used by an object itself. This is useful for comparing the container overhead of generators and lists.

## Task

Implement `footprint_ratio(N)`:

```python
def footprint_ratio(N: int) -> float:
    ...
```

The function must create:

```python
[x for x in range(N)]
```

and:

```python
(x for x in range(N))
```

and return:

$$
\frac{\operatorname{sys.getsizeof}([x \text{ for } x \text{ in range}(N)])}
{\operatorname{sys.getsizeof}(x \text{ for } x \text{ in range}(N))}
$$

Return the value as a Python `float`.

Do not materialize the generator by converting it to a list. The comparison is about the memory footprint of the list object versus the live generator object.

## Example

```python
ratio = footprint_ratio(1000)

# The list comprehension stores many elements.
# The generator expression stores only iteration state.
# ratio is greater than 1 on CPython.
```

## What the gate checks

The gate uses the real CPython `sys.getsizeof` implementation as the oracle. It evaluates several values of $N$ and compares the returned ratios against ratios computed from freshly created list comprehensions and generator expressions.

The score is the average agreement with the CPython measurements. A score of at least $0.99$ is required, so implementations that return an inverse ratio or measure the wrong objects fail.
