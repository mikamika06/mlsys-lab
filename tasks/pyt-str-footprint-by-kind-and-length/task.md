## Context

CPython strings use a flexible internal representation. The representation stores
characters using one of several element widths depending on the largest code point
in the string.

For a string of length $n$, the payload contribution is related to the storage
kind $k$:

$$
\mathrm{payload}(n) \approx k \cdot n .
$$

ASCII strings have a specialized compact layout, while other strings use wider
elements when their maximum code point requires them. The final memory footprint
reported by `sys.getsizeof` also includes object headers and implementation
details.

The exact size is therefore determined by CPython's internal string layout:

$$
\mathrm{sizeof}(s) = \mathrm{header}(s) + \mathrm{payload}(s).
$$

Different strings with the same length can have different footprints because their
maximum code points can select different internal kinds.

## Task

Implement `str_footprint(lengths, max_codepoints)`:

```python
def str_footprint(lengths, max_codepoints):
    ...
```

The two arguments are equal-length sequences. Each pair
`(lengths[i], max_codepoints[i])` describes a string of that length whose largest
code point is `max_codepoints[i]`.

Return a list containing the exact value that `sys.getsizeof` reports for each
constructed string.

The implementation should model CPython 3.12 string footprint behavior rather
than measuring unrelated containers.

## Example

```python
sizes = str_footprint([3, 2, 1], [65, 233, 128512])
# The values are the CPython 3.12 sys.getsizeof results for:
# "AAA", "éé", and "😀"
```

## What the gate checks

The gate constructs real Python strings and uses CPython's `sys.getsizeof`
implementation as the oracle. Your returned list must exactly match those
measurements for ASCII, Latin-1, BMP, and non-BMP strings.

The metric is `exact_match`. A result passes only when every predicted footprint
matches the oracle output.
