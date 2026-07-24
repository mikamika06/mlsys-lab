## Context

`double` needs 8-byte alignment; `bool` needs only 1. Whenever a `bool`
sits directly before a `double` in a struct, the compiler must insert
enough padding to push that `double` back onto an 8-byte boundary — and it
does this at *every* such gap, independently.

```cpp
struct BadRecord {
    double a;      // offset 0,  8 bytes
    bool   flag1;  // offset 8,  1 byte
    double b;      // offset 16 (7 bytes of padding inserted before it!)
    bool   flag2;  // offset 24, 1 byte
};               // sizeof == 32 (7 more tail bytes, to round up to 8)
```

18 bytes of real data, 32 bytes of struct — 14 bytes (44%) is pure padding,
because the two 1-byte bools each individually break the 8-byte alignment
stream of the doubles around them.

## Task

`sol.hpp` declares the broken layout above. Your starting point in
`solve.cpp` *is* `BadRecord`, reporting its own (broken) `sizeof`/`offsetof`
through:

```cpp
size_t record_size();   // sizeof(your record)
size_t offset_a();      // offsetof(your record, a)
size_t offset_b();      // offsetof(your record, b)
size_t offset_flag1();  // offsetof(your record, flag1)
size_t offset_flag2();  // offsetof(your record, flag2)
```

Fix it: define your own struct with the same 4 fields (two `double`, two
`bool`), reordered so the two doubles are adjacent and the two bools are
adjacent — grouping every 8-byte-aligned field together leaves only ONE
rounding gap, at the very end, instead of one after every bool. Update the
five functions above to report `sizeof`/`offsetof` against your reordered
struct.

The driver (`main.cpp`, fixed) prints `record_size`, all four offsets, and
`size_ratio` = `record_size() / 32.0` (32 being the broken layout's fixed
size).

## Example

Grouping doubles first, then bools:

```cpp
struct FixedRecord {
    double a;      // offset 0
    double b;      // offset 8
    bool   flag1;  // offset 16
    bool   flag2;  // offset 17
};               // sizeof == 24 (6 tail bytes to round up to 8)
```

```
record_size=24
offset_a=0
offset_b=8
offset_flag1=16
offset_flag2=17
size_ratio=0.750000
```

25% smaller than the broken layout, from reordering alone — no field was
added, removed, or resized. The unmodified `BadRecord` prints
`record_size=32` and `size_ratio=1.000000`.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed number (including `size_ratio`) to
match the same driver linked against the reference layout within
`max_abs_err <= 1e-6`. Leaving `BadRecord` untouched reports
`size_ratio=1.000000` and offsets `0/16/8/24` instead of `0/8/16/17` —
both the ratio and every offset fail the gate until the doubles and bools
are actually regrouped.
