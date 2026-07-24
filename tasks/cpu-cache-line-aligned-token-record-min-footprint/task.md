## Context

A "token" record has three fields a hot loop touches on every pass — `id`
(read), `count` (read-modify-write), `flags` (read-modify-write) — and two
fields that exist on the record but are only ever read by a rare, separate
path: a `name[24]` buffer and a `ts` timestamp. How you *order* those five
fields inside the struct changes `sizeof(record)` through alignment padding
alone, with zero change to what the record logically stores.

A 64-byte cache line can hold several small records at once, but only if
records are small enough and their hot bytes aren't scattered by padding.
Interleave hot and cold fields carelessly, or tack on unnecessary padding,
and `sizeof(record)` balloons — inflating the *total* bytes the hot loop's
working set occupies. Once that working set no longer fits in the cache,
every sweep over the array re-misses almost everything, even though the
hot loop still only ever reads 9 useful bytes per record.

## Task

Implement, in your `.cpp` file, ONE struct `TokenRecord` with real C++
member declarations for:

```cpp
uint32_t id;      // HOT
uint32_t count;   // HOT
uint8_t  flags;   // HOT
char     name[24];// COLD
uint64_t ts;       // COLD
```

Order and pack the fields so the 3 HOT fields sit adjacent, with no
alignment gaps between them, and the whole struct is as small as the
compiler's own alignment rules allow — don't add any padding of your own.
Report your layout by implementing:

```cpp
size_t record_size();   // sizeof(TokenRecord)
size_t offset_id();     // offsetof(TokenRecord, id)
size_t offset_count();  // offsetof(TokenRecord, count)
size_t offset_flags();  // offsetof(TokenRecord, flags)
size_t offset_name();   // offsetof(TokenRecord, name)
size_t offset_ts();     // offsetof(TokenRecord, ts)
```

The driver (`main.cpp`, fixed) allocates a 64-byte-aligned array of $N = 70$
records using your `record_size()` and offsets, initializes every field,
then runs $40$ measured passes over the array that touch only `id`,
`count`, `flags` per record (through a fixed fully-associative, 64-line x
64-byte-line LRU cache model declared in `sol.hpp`), and prints
`record_size`, the total miss count, and a checksum of the values read.

## Example

The natural packing —

```cpp
struct TokenRecord {
    uint32_t id;
    uint32_t count;
    uint8_t  flags;
    char     name[24];
    uint64_t ts;
};
```

— gives `sizeof(TokenRecord) == 48` (`id`+`count`+`flags` back-to-back at
offsets 0/4/8, then `ts` needs 8-byte alignment so `name` is followed by 7
bytes of unavoidable padding before it). $70 \times 48 = 3360$ bytes fits
comfortably inside the 4096-byte cache, so only the first pass is cold:

```
record_size=48
misses=52
checksum=155400
```

Reordering to `name, id, ts, count, flags` and tacking on a `char pad[19]`
"for future use" — a layout that still stores the exact same 5 logical
fields — pads `sizeof(TokenRecord)` up to 64. $70 \times 64 = 4480$ bytes no
longer fits in the 4096-byte cache, so the streaming access pattern re-misses
on essentially every touch of every pass:

```
record_size=64
misses=2800
checksum=155400
```

Same checksum (the logical data and the arithmetic are identical either
way) — 53.8x more misses purely from wasted bytes in the record layout.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires the three printed numbers to `exact_match` the same
driver linked against the reference layout. Getting the checksum right but
choosing a bloated or hot/cold-interleaved layout leaves `record_size` and
`misses` wrong and fails the gate — the padding cost has to disappear from
the printed numbers, not just leave the (layout-independent) checksum
unchanged. The starter returns `0` from every function, which collapses
every record to a single aliased address and prints numbers nothing like
the reference.
