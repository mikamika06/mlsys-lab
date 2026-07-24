## Context

CPython allocates many small objects through its internal allocator. A request for
$n$ bytes may occupy a larger resident block because the allocator stores objects
in size classes. The wasted space inside an allocated block is internal
fragmentation.

For an allocation pattern $P$ containing requested byte counts $p_i$, define the
requested total as

$$R = \sum_i p_i .$$

The resident total can be approximated using the real CPython object footprint for
each allocation. For a concrete allocation object $x_i$, let $s_i$ be
`sys.getsizeof(x_i)`. The resident total is

$$S = \sum_i s_i .$$

The fragmentation ratio is

$$F = \frac{S}{R} .$$

A ratio of $1$ would mean no overhead. Values above $1$ indicate additional
resident bytes compared with the requested payload.

## Task

Implement `internal_fragmentation_ratio(pattern)`:

```python
def internal_fragmentation_ratio(pattern):
    ...
```

`pattern` is an iterable of non-negative integers. Each integer is a requested
byte count. The function must allocate a `bytearray` of each requested size,
measure its CPython object size with `sys.getsizeof`, and return the resident
bytes divided by requested bytes.

Return a Python `float`. The denominator is the sum of the requested sizes. The
function should raise `ValueError` if the pattern has no bytes requested.

## Example

```python
print(internal_fragmentation_ratio([8, 32, 64]))
```

The exact value depends on the running CPython build because object headers and
allocator details differ between versions. The result is computed from the active
interpreter's `sys.getsizeof` values.

## What the gate checks

The gate builds several allocation patterns and computes the expected ratio using
the real CPython object-size oracle. The returned value must match the oracle
within the required exact floating-point comparison. Implementations that use a
fixed header estimate or only sum the requested bytes will fail because they do
not follow the active interpreter's allocation behavior.
