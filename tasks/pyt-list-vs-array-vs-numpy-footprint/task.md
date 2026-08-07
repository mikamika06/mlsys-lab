## Context

Python containers have different memory layouts. A Python `list` stores references to
objects and may allocate extra capacity to make appending efficient. A typed
`array('q')` stores signed 64-bit integers directly, while a Python `int64` array
stores a contiguous numeric buffer.

For a sequence of $N$ integers, the footprint ratio between two representations
is

$$R_{A,B} = \frac{\mathrm{bytes}(A)}{\mathrm{bytes}(B)}.$$

The exact size of a Python list depends on the CPython runtime because the
interpreter manages object headers and over-allocation internally. Therefore the
measurement must come from the runtime instead of a manually derived formula.

## Task

Implement `footprint_ratios(n)`.

The function receives a positive integer $n$ and returns a dictionary with these
floating-point entries:

- `"list_vs_array"`: `sys.getsizeof(list(range(n))) / sys.getsizeof(array("q", range(n)))`
- `"list_vs_list"`: `sys.getsizeof(list(range(n))) / sys.getsizeof(my_list)`, where the list is created with `list(range(n))`
- `"array_vs_numpy"`: `sys.getsizeof(array("q", range(n))) / numpy_array.nbytes`

Use real measurements from Python objects. Do not hardcode CPython headers,
allocation growth factors, or element sizes.

## Example

```python
result = footprint_ratios(1000)

# Example shape:
# {
#   "list_vs_array": 2.9,
#   "list_vs_numpy": 2.9,
#   "array_vs_numpy": 1.0
# }
```

The exact values depend on the interpreter and platform.

## What the gate checks

The grader creates several values of $n$ and computes the expected ratios using
the runtime oracle: `sys.getsizeof` for Python objects and Python for the numeric
buffer size.

The returned values must match the oracle within an absolute error of
$10^{-12}$. Solutions that assume every representation uses exactly $8N$ bytes
will fail because Python lists contain interpreter-managed overhead and spare
capacity.
