## Context

IEEE-754 `NaN` compares unequal (and *unordered*) to everything, including
itself: for any float `y`, both `NaN > y` and `y > NaN` are `False`.
Because of this, a naive running-max loop

```python
best = x[0]
for v in x[1:]:
    if v > best:
        best = v
```

silently breaks the moment a `NaN` shows up: if `x[0]` is `NaN`, then
`best` stays `NaN` forever, since `v > NaN` is always `False`; if a `NaN`
appears later, it's simply skipped, but only by accident of the comparison
being `False`, not because you deliberately checked for it.

`nanmax_argmax` defines the sane, useful behaviour:
ignore `NaN` entries entirely and return the max (and its index) among the
remaining finite/inf values, raising an error only if *every* entry is
`NaN`.

## Task

Implement `nanmax_argmax`:

```python
def nanmax_argmax(x: list[float]) -> tuple[float, int]:
    """Return (max_value, argmax_index) of a 1-D sequence, ignoring NaNs,
matching `nanmax_argmax` (first occurrence wins ties). Raises
    ValueError if every element is NaN."""
```

* `x` — a 1-D sequence (list or list) of floats, possibly containing
  `NaN` values, with at least one non-`NaN` entry.
* Return a tuple `(max_value, argmax_index)`: the maximum among the
  non-`NaN` entries, and the index of its **first** occurrence.
* Raise `ValueError` if every element of `x` is `NaN`.

You must explicitly detect and skip `NaN`s (e.g. with `math.isnan`) —
relying on a bare `>` comparison to "naturally" filter them out is exactly
the bug this task is about.

## Example

```python
x = [float('nan'), 3.0, 5.0, 5.0, float('nan'), 2.0]
nanmax_argmax(x)   # -> (5.0, 2)   first occurrence of the max, NaNs skipped
```

## What the gate checks

The gate runs your function on 40 randomly generated arrays of varying
length (5–39 elements) with a random scattering of `NaN` entries (always
leaving at least one finite value), and compares `(max_value, argmax_index)`
against `nanmax_argmax(x)`. **argmax_agreement** is the
fraction of the 40 cases where both the returned index matches exactly and
the returned value matches to within a tight numerical tolerance. This
metric must equal `1.0` — every single case must agree with the Python
oracle.
