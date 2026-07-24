## Context

A struct's fields don't sit flush against each other in memory. Each
field must start at a byte offset that's a multiple of its own
alignment, so the compiler inserts **inter-field padding** before a field
whenever the previous field didn't end on that boundary. And once the
last field is placed, the struct's overall size gets rounded up to a
multiple of the struct's own alignment — the largest alignment among its
fields — so that when you pack many of these structs back-to-back in an
array, every element (not just the first) starts properly aligned. That
final rounding is **tail padding**.

Both kinds of padding are pure overhead: bytes that hold no data but
still have to be allocated and, when the array is scanned, still get
pulled through the cache. For an array of `count` records, the total
waste is `count` times the per-record padding — reordering fields to
shrink that per-record padding (e.g. sorting fields from largest to
smallest alignment) is a common no-cost win for large arrays.

## Task

Implement

```cpp
long total_padding_bytes(const int* field_sizes, const int* field_aligns, int num_fields, long count);
```

which computes the struct's laid-out size by walking its `num_fields`
fields in order: keep a running `offset` (starting at 0) and the running
`max_align` (starting at 1); for each field `i`, advance `offset` up to
the next multiple of `field_aligns[i]` if it isn't already there (that
gap is inter-field padding), then add `field_sizes[i]`; after the last
field, round `offset` up to the next multiple of `max_align` (tail
padding) to get the padded record size. Return

```
(padded_record_size - sum_of_field_sizes) * count
```

## Example

Fields `{char(1,align 1), int(4,align 4), char(1,align 1), double(8,align 8)}`:
offset 0 → `char` ends at 1 → pad 3 to align `int` at 4, ends at 8 →
`char` at 8, ends at 9 → pad 7 to align `double` at 16, ends at 24 →
`max_align = 8`, `24` is already a multiple of 8, so the padded record
size is 24. Sum of field sizes is `1+4+1+8 = 14`, so each record wastes
`10` bytes; for `count = 10` records, `total_padding_bytes` returns `100`.

## What the gate checks

The driver runs 5 scenarios built from real C++ structs — the field
`size`/`align` arrays your function receives are read via `sizeof`/
`alignof` off each struct's actual members, in declared order — and
prints the total padding your function computes for each at a fixed
record count. The grader compiles `solve.cpp` with `clang++ -O2
-std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{all 5 printed totals match the reference}
$$

The reference's per-record padding for each scenario equals
`sizeof(struct) - sum_of_field_sizes` for the corresponding real struct —
that's the compiler's own layout, not a guessed number. A stub that
returns `0` reports no padding at all for every scenario and fails
immediately; forgetting the final tail-padding rounding (only handling
inter-field gaps) under-counts every scenario whose largest-alignment
field isn't last.
