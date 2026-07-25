## Context

"AoS vs SoA" is usually taught as a rule of thumb: array-of-structs is
better when you touch whole records, struct-of-arrays is better when you
only touch a few fields across many records. The rule is right, but it's
a *consequence* of cache-line counting, not an axiom — and the crossover
point depends on the actual numbers (field count, field sizes, how many
fields a pattern touches), not just "few fields good, many fields bad."

Under AoS, a cache line pulls in several whole records at once — so
reading even one field of every record still has to fetch every line the
records span, exactly as if every field were read. Under SoA, each field
lives in its own array, so a pattern that only touches `K` of `F` fields
only ever pulls in `K` fields' worth of lines — but summing `K` separate
arrays' rounding-up-to-a-whole-line overhead can, once `K` gets close to
`F`, actually cost *more* lines than one contiguous AoS sweep would.

## Task

Implement

```cpp
int soa_is_optimal(int N, int F, const int* field_bytes, const bool* mask);
```

`N` records each have `F` fields of byte sizes `field_bytes[0..F)`. The
access pattern reads every record `r` in `[0, N)`, and for each field `f`
with `mask[f]` true, reads that field's bytes for record `r`. Model both
layouts:

- **AoS**: fields packed back-to-back per record (record size = sum of
  `field_bytes`; field `f` of record `r` sits at byte offset
  `r*record_bytes + sum(field_bytes[0..f))`).
- **SoA**: each field gets its own contiguous array of `N` elements, at a
  fresh base address padded up to a whole number of 64-byte lines (so no
  two fields' arrays ever share a line).

Count the distinct 64-byte lines the pattern touches under each layout
(a field access that straddles a line boundary touches every line it
overlaps). Return `1` if SoA's line count is `<=` AoS's, else `0` (AoS
strictly wins).

## Example

1000 records of four 4-byte fields (`record_bytes = 16`), reading just
field 0: AoS still has to fetch every line the 1000 records span —
`ceil(1000*16/64) = 250` lines, exactly as many as reading all 4 fields
would need — while SoA only fetches field 0's own array:
`ceil(1000*4/64) = 63` lines. SoA wins by 4x.

## What the gate checks

The driver classifies 5 fixed access patterns over that same 1000-record,
4-field layout — reading 1, 2, 3, then all 4 of the fields — plus a 5th
pattern over 500 records of 5 fields, reading 2 non-adjacent ones. It
prints all 5 labels. The grader compiles `solve.cpp` with `clang++ -O2
-std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{all 5 printed labels match the reference}
$$

The reference prints `labels=1,1,1,0,1`: reading 1, 2, or 3 of 4 fields
favors SoA every time, but reading all 4 fields flips it — AoS's `250`
lines beats SoA's `4 * 63 = 252`, because summing four independently
rounded-up field arrays costs slightly more than one contiguous sweep. A
classifier that always answers "SoA" (the naive rule of thumb, ignoring
the actual line counts) gets the 4th pattern wrong and fails the gate.
