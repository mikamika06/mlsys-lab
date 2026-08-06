## Context

Python objects can store instance state in different layouts. A normal class instance commonly has a `__dict__`, while a class using `__slots__` stores attributes directly in fixed slot descriptors. A `namedtuple` stores values in a tuple-like layout. CPython can also share dictionary keys between instances of the same class, reducing the per-instance dictionary overhead.

For an instance with footprint $S$ bytes and a reference slot footprint $S_{\mathrm{slot}}$ bytes, a relative footprint can be expressed as

$$r = \frac{S}{S_{\mathrm{slot}}}.$$

This task measures how these layouts scale as the number of stored fields changes.

## Task

Implement `footprint_sweep(widths)`:

```python
def footprint_sweep(widths):
    ...
```

`widths` is an iterable of positive integers. Each integer is the number of attributes stored in an instance.

Return a list of shape $(n, 3)$ with `float64` values. For each width, the columns must be:

1. ratio of a normal dict-backed class instance footprint to a slots instance footprint,
2. ratio of a `namedtuple` instance footprint to a slots instance footprint,
3. ratio of a key-shared class instance footprint to a slots instance footprint.

The footprint should be measured using CPython object sizes. The function should perform the sweep rather than returning a fixed table.

## Example

```python

ratios = footprint_sweep([1, 4, 16])

# rows correspond to widths 1, 4, and 16
# columns correspond to:
# [dict_class / slots, namedtuple / slots, key_shared / slots]
assert ratios.shape == (3, 3)
assert ratios.dtype == float
```

## What the gate checks

The gate creates the same layouts using the running CPython interpreter and measures them with `sys.getsizeof`. The returned curve is compared to this runtime oracle.

The maximum elementwise relative deviation from the oracle curve must be small:

$$\max_i \frac{|x_i - y_i|}{|y_i| + 10^{-12}} \le 10^{-3}.$$

The gate metric reports the corresponding score

$$\frac{1}{1+\mathrm{max\_relative\_deviation}},$$

which must be at least $0.999$.
