## Context

C++ offers two fundamental data-structure layouts for collections of records.

**Array of Structures (AoS):** each element of the array is a full struct
containing all fields of one record, laid out contiguously — every field of
one struct instance is a plain scalar member.

**Structure of Arrays (SoA):** the top-level struct holds a separate
*array* for each logical field. Element $i$ of field $j$ lives at a
predictable offset with no interleaving from other fields, giving regular
stride access and better spatial locality when only a subset of fields is
hot.

Key concepts:

- **Cache-line utilisation.** AoS fetches irrelevant fields on every touch;
  SoA keeps fetched bytes on-topic for the query pattern.
- **False sharing.** In multithreaded code, AoS records that straddle a
  cache-line boundary force coherency traffic between cores. SoA partitions
  by field, so threads touching different fields rarely contend.
- **The signal that actually distinguishes them.** It is not whether the
  fields happen to share a type — a `{float r, g, b, a}` color record is
  still one interleaved AoS record, even though all four fields are
  `float`. The real signal is whether a field IS an array (a whole parallel
  column): if even one field is, the struct is a column store (SoA); if
  every field is a plain per-record scalar, it's a row store (AoS).

## Task

Implement `classify_layout` in `solve.cpp`:

```cpp
int classify_layout(const Field* fields, int nfields);
```

`Field` (declared in `sol.hpp`) is `{const char* type; bool is_array;}`.
Return `1` (SoA) if any field has `is_array == true`; otherwise return `0`
(AoS), including for `nfields == 0`.

The fixed driver in `main.cpp` runs your function over 10 fixed struct
layouts and prints the 10 labels.

## Example

```cpp
Field vec3[] = {{"float", false}, {"float", false}, {"float", false}};
classify_layout(vec3, 3);   // -> 0 (AoS): three plain scalar floats

Field soa[] = {{"float", true}, {"float", true}, {"float", true}};
classify_layout(soa, 3);    // -> 1 (SoA): three parallel arrays
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across all 10 layouts, 4 of which
contain at least one array field. The starter labels everything `0`
(AoS), so it fails on every layout that actually contains an array field.
