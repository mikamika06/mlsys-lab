## Context

A compiler places each struct field at an offset that's a multiple of the
field's own alignment (usually its size, for the fundamental types).
Wherever the next field's alignment demand isn't already met, the
compiler inserts invisible padding bytes to skip ahead. The struct's total
size is then rounded up to a multiple of its largest field's alignment,
so instances tile cleanly in an array. None of this depends on *which*
fields you declare — only on the *order* you declare them in. The classic
fix, with zero change in memory semantics, is to sort fields from largest
alignment to smallest: pack all the 8-byte-aligned fields together first,
then the 4-byte ones, then 2-byte, then the 1-byte fields last, so nothing
after the first field is ever waiting on an alignment boundary it hasn't
already reached.

## Task

`NaiveStruct` (in `sol.hpp`) declares one `bool`, one `double`, one `char`,
one `int32_t`, a second `bool`, one `int64_t`, and one `int16_t`, in a
deliberately scattered order. Implement

```cpp
size_t packed_struct_size();
```

by defining your own struct with the *same 7 field types* (any order you
like, any field names), and returning `sizeof` it — chosen to minimize
that size under the compiler's normal alignment rules (no packing
attributes).

## Example

`NaiveStruct`'s declared order forces `value_x` (8-byte aligned) to start
at offset 8 (skipping 7 bytes after `flag_a`), `id` (8-byte aligned) to
start at offset 32 (skipping 7 more after `flag_b`), and rounds the whole
struct up to `48` bytes. Grouping by descending alignment — `double`,
`int64_t`, `int32_t`, `int16_t`, then the two `bool`s and the `char` —
leaves no gaps at all until the very end: 8+8+4+2+1+1+1 = 25 bytes of
actual data, rounded up to the next multiple of 8, giving `32` bytes.

## What the gate checks

`exact_match` on `(sizeof(NaiveStruct), packed_struct_size())` — reference
values `(48, 32)`. Any field ordering that reaches the true minimum of 32
bytes passes, since the gate only compares the final size, not which
order you picked; a starter that echoes `NaiveStruct`'s own layout (or
any ordering that leaves avoidable padding) prints a larger second
number and fails the match.
