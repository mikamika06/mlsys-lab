## Context

A Python dictionary uses a hash table internally. Insertions, deletions, and resizing
can change the memory footprint reported by `sys.getsizeof`.

When keys are removed, a dictionary may retain internal table capacity instead of
immediately shrinking. Later insertions can reuse space or trigger a resize. The
footprint of a dictionary can therefore depend on its mutation history, not only on
the number of live entries.

For a dictionary footprint measurement, define the size ratio as

$$
r = \frac{S_{\mathrm{after}}}{S_{\mathrm{before}}},
$$

where $S_{\mathrm{before}}$ and $S_{\mathrm{after}}$ are the byte sizes reported by
`sys.getsizeof` before and after a sequence of insert and delete operations.

## Task

Implement `dict_footprint_churn(n, cycles)`:

```python
def dict_footprint_churn(n: int, cycles: int) -> tuple[int, int]:
    ...
```

Create a dictionary containing `n` integer key-value pairs. Record its initial
footprint with `sys.getsizeof`. Then perform `cycles` rounds where all current
entries are deleted and `n` new integer key-value pairs are inserted. Record the
final footprint with `sys.getsizeof`.

Return a tuple:

```text
(before_size, after_size)
```

Both values must be the integer byte counts returned by `sys.getsizeof`. Use integer
dictionary keys and values only.

## Example

```python
before, after = dict_footprint_churn(1000, 5)

# before and after are integer byte counts from sys.getsizeof
print(before > 0)
print(after > 0)
```

## What the gate checks

The gate runs the implementation against a real CPython dictionary oracle using the
same mutation sequence and compares the returned footprint ratio

$$
\frac{S_{\mathrm{after}}}{S_{\mathrm{before}}}
$$

with the oracle ratio. The result passes when the measured ratio matches the CPython
`sys.getsizeof` measurement for all tested cases.
