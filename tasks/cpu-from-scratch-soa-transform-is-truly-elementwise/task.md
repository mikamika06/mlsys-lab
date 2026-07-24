## Context

Array-of-Structs (AoS) packs every field of a record together:
`x0,y0,z0,w0, x1,y1,z1,w1, ...`. Struct-of-Arrays (SoA) instead packs
each field into its own contiguous array: `x0,x1,x2,..., y0,y1,y2,...`.
Converting one field of an AoS dataset into a standalone SoA array is
conceptually trivial — copy `field[i]` for every `i` — but it's easy to
accidentally do MORE work than that: touching a whole AoS record when you
only need one of its fields, or re-reading the same address more than
once, both burn real memory bandwidth for no reason. A **genuinely
elementwise** transform touches *exactly* one source byte-group and one
destination byte-group per element — nothing extra.

## Task

Implement

```cpp
void aos_field_to_soa(long aos_base, long soa_out_base, int record_count,
                       int field_count, int field_index);
```

which extracts field `field_index` (0-based) from every one of
`record_count` AoS records — `field_count` 4-byte float fields each,
record `i` starting at `aos_base + i * field_count * 4` — into a fresh
contiguous SoA output array at `soa_out_base` (element `i` at
`soa_out_base + i * 4`). For every `i` in `[0, record_count)`, in
increasing order, call `touch()` (declared in `sol.hpp`) on the source
field address, then on the destination address — **exactly once each**.

$$
\text{src}(i) = \text{aos\_base} + i \cdot \text{field\_count} \cdot 4 + \text{field\_index} \cdot 4, \qquad
\text{dst}(i) = \text{soa\_out\_base} + i \cdot 4
$$

## Example

`field_count=4, field_index=2, aos_base=0`: record `5`'s field `2` lives
at `5*4*4 + 2*4 = 88`. If `record_count=2000`, `soa_out_base = 2000*4*4 =
32000`, so the output element for record `5` lives at `32000 + 5*4 =
32020`. A truly elementwise transform of all 2000 records touches exactly
`2000` source addresses and `2000` destination addresses — `4000` touches
total, never more.

## What the gate checks

`main.cpp` runs `aos_field_to_soa` over 4 fixed scenarios — 4-field
records extracting the first field, the same records extracting the last
field, wider 8-field records extracting a middle field, and narrow
2-field records — each against a fresh 2048-byte (64-byte line, 8-set,
4-way) cache model, and prints the touch count, the miss count, and
whether `touch_count == 2 * record_count` for each. The candidate's full
stdout is compared byte-for-byte (`exact_match = 1.0`) against the
reference's. Touching an extra field per record (e.g. reading all
`field_count` fields "just in case"), touching the destination twice, or
skipping records, all inflate or shrink the touch count away from exactly
`2 * record_count` and away from the reference's miss count, failing
every one of the 4 scenarios at once.
