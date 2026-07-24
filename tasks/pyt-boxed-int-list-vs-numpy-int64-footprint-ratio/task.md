## Context

A Python `list` of integers stores an array of **pointers**; each `int` is a
separate boxed heap object (a `PyObject` header plus a variable-length "digit"
array for the magnitude — `sys.getsizeof` grows roughly every 30 bits of
magnitude, so `sys.getsizeof(2**29)` and `sys.getsizeof(2**31)` are not the
same). CPython also permanently caches and shares the small integers
$-5 \le v \le 256$: **every** reference to one of these values, however it was
constructed, points at the *same* singleton object. So if a list references a
cached small int $k$ times, those $k$ pointers cost list-container bytes each,
but the pointed-to `int` object itself is only allocated **once** — counting
it $k$ times overstates the list's real heap footprint.

A NumPy `int64` array, by contrast, stores the values inline in one flat
buffer, so its total footprint is just `sys.getsizeof` of the array object
(buffer + fixed array-object header).

## Task

Implement `list_footprint_ratio`:

```python
def list_footprint_ratio(values: list) -> float:
    ...
```

`values` is a Python `list` of `int` objects (possibly containing repeated
*references* to the same cached small-int object). Compute:

1. **list footprint** = `sys.getsizeof(values)` (the container) **plus** the
   sum of `sys.getsizeof(v)` over the *distinct objects* referenced by the
   list — use `id(v)` to deduplicate, so an object referenced multiple times
   (e.g. a shared small int) is only counted once.
2. **array footprint** = `sys.getsizeof(np.array(values, dtype=np.int64))`.

Return `list footprint / array footprint` as a `float`. Use only real runtime
measurements (`sys.getsizeof`, `id`) — do not hardcode CPython's object-header
size, the small-int cache bounds, or the per-digit growth step; all of these
can be read off the actual objects.

## Example

```python
import sys
x = 100          # in CPython's small-int cache
values = [x] * 5 # 5 pointers, but ONE underlying int(100) object
list_footprint_ratio(values)
# == sys.getsizeof(values) + sys.getsizeof(x)     (x counted once, not 5x)
#    ---------------------------------------------
#    sys.getsizeof(np.array(values, dtype=np.int64))
```

## What the gate checks

`rel_err` — the grader builds several fixture lists: a mix of every cached
small int $-5..256$ (each appearing once) plus 20 extra references to the
*same* cached value `100`, followed by 500 distinct large ints at a randomly
shifted offset (repeated across 3 differently-seeded cases), plus edge cases
(`[]`, `[0]`, and a list mixing very large magnitudes up to `2**62` that push
`sys.getsizeof` past several digit-growth steps). Your ratio is compared to
one computed the same way from real `sys.getsizeof`/`id` measurements; the
worst-case relative error across all fixtures must satisfy `rel_err < 1e-9`.
Naively summing `sys.getsizeof(v)` over every list *position* (double-counting
the shared `100` references) or using `arr.nbytes` instead of
`sys.getsizeof(arr)` for the array side both produce a measurably wrong ratio.
