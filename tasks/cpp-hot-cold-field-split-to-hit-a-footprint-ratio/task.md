## Context

Iterating over an array of a large `Entity` struct, but only touching two or
three of its fields inside the loop, wastes cache bandwidth: every element
you stream past drags its unused fields through the cache too. The classic
fix is a **hot/cold split** — pull the frequently-touched fields into a
small `HotEntity` and leave everything else in a `ColdEntity`, so the tight
loop only ever streams the compact one.

Splitting fields into two groups isn't enough by itself, though: within the
new `HotEntity`, **field order still determines padding**. C/C++ places each
field at the next offset that is a multiple of its own alignment (which, for
plain `char`/`short`/`int`/`double`/pointer fields, equals its size), and
rounds the whole struct's size up to a multiple of its largest field's size.
Declaring a `HotEntity` in a padding-unfriendly order can waste as much space
as leaving the cold fields in.

## Task

Implement

```cpp
int split_struct(const int* fields, const int* is_hot, int n,
                  int* hot_out, int* cold_out);
```

- Partition `fields[0..n)` by `is_hot[i]`: hot sizes go into `hot_out`, cold
  sizes go into `cold_out` (cold in original relative order — its padding
  isn't graded).
- **Reorder** `hot_out` to minimize `struct_size(hot_out, hot_count)`
  (declared in `sol.hpp`, defined in `main.cpp`, applying the layout rule
  described above). Sorting the hot sizes in **descending** order is optimal
  here, since every field kind used (`char`=1, `short`=2, `int`=4,
  `double`/pointer=8) has alignment equal to its size.
- Return `hot_count`, the number of hot fields.

## Example

Hot field sizes `{1, 8, 1, 4, 2, 1}` (char, double, char, int, short, char),
laid out in that original order, pad out to 32 bytes (a `double` right after
a `char` forces 7 bytes of padding). Sorted descending —
`{8, 4, 2, 1, 1, 1}` — the same six fields pack into 24 bytes: no field is
ever placed at an offset its own alignment doesn't already satisfy.

## What the gate checks

The driver runs two fat `Entity` layouts (12 and 8 fields, hot and cold
interleaved in a deliberately padding-unfriendly order) and prints, for
each: the original struct's footprint, the hot fields (count and reordered
sizes) with their resulting footprint, and the cold fields (count and sizes)
with theirs. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed count, field list, and footprint matches the reference}
$$

Getting the hot/cold *partition* right but leaving the hot fields in their
original order still fails: the footprint printed after the split would be
larger than the reference's minimally-padded one, even though the field
*set* is correct.
