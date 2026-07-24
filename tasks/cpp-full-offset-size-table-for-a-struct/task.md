## Context

A C++ compiler lays out a struct's fields sequentially in memory. To keep access fast, it inserts **padding** bytes so each field starts at a multiple of its own alignment requirement, then adds **tail padding** so the whole struct's size is a multiple of its largest field's alignment (so arrays of the struct stay tightly packed).

On this platform's LP64 ABI:
- `bool`, `char` = 1 byte
- `short` = 2 bytes
- `int`, `float` = 4 bytes
- `long`, `long long`, `double`, and any pointer = 8 bytes

The alignment requirement of every one of these basic types equals its size.

## Task

Implement:

```cpp
void struct_layout(const FieldType* fields, int n, FieldLayout* out, int* total_size_out);
```

For a struct whose members appear, in order, with the types in `fields`, fill `out[i] = {offset, size}` for each member, and write the struct's total size (including tail padding) to `*total_size_out`.

## Example

For `fields = {Char, Int, Double}`: `out = [{0,1}, {4,4}, {8,8}]` and `*total_size_out = 16` — `char` at offset `0`, `3` padding bytes so `int` lands on a 4-byte boundary at offset `4`, then `double` naturally falls on an 8-byte boundary at offset `8`, and the struct's own size is already a multiple of `8` (its largest alignment), so there is no tail padding beyond that.

## What the gate checks

`main.cpp` defines ten real structs, one per test field list, and reads each one's *true* layout straight off the compiler via `offsetof`/`sizeof` — not a simulator. It calls `struct_layout` with the matching `FieldType` sequence and prints the candidate's predicted offsets, sizes, and total, plus whether they match the compiler's real values. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Ignoring alignment (packing fields with no padding at all) matches only structs that happen to need none, such as an all-`char` struct, and diverges everywhere padding actually matters.
