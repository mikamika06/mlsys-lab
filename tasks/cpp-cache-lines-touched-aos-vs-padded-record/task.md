## Context

When iterating over an Array of Structs (AoS) and reading a single field, the
CPU fetches memory in 64-byte chunks called cache lines. If several structs
fit in one line, one fetch serves several reads. If a struct is inflated with
padding until it fills — or exceeds — a full line, every record needs its own
fetch, and the useful field is a tiny fraction of what gets pulled in.

Two fixed records here, laid out by the real compiler (`sol.hpp`):

```cpp
struct Compact { int id; double value; char flag; };   // sizeof == 24
struct Padded  { int id; double value; char flag; char pad[40]; }; // sizeof == 64
```

`Compact` packs 4-byte `id`, 4 bytes of alignment padding, 8-byte `value`,
1-byte `flag`, then 7 bytes of tail padding (to round up to the struct's
8-byte alignment) — 24 bytes total, so 64/24 records don't even split evenly:
a run of `Compact` records shares cache lines across the *record* boundary.
`Padded` adds an oversized `char pad[40]` tail so the whole record, including
its own tail padding, is exactly 64 bytes — one full cache line per record,
no sharing, no benefit from the reserved space.

## Task

The driver (`main.cpp`, fixed) allocates a 64-byte-aligned array of 1000
`Compact` and a 64-byte-aligned array of 1000 `Padded`, fills both with the
same deterministic `value` sequence, and gives you a cache-line probe
declared in `sol.hpp`:

- `cache_reset()` clears the set of touched lines.
- `touch(p)` records the 64-byte line containing address `p`.
- `lines_touched()` returns how many distinct lines have been touched since
  the last reset.

Implement, in `solve.cpp`:

- `sum_value_compact(const Compact* arr, int n)` — sum `arr[i].value` over
  all `n` elements. Call `touch(&arr[i])` once per element: reading any
  field of `arr[i]` pulls in the cache line(s) covering that record.
- `sum_value_padded(const Padded* arr, int n)` — same sweep over `Padded`.
  Call `touch(&arr[i])` once per element.

## Example

With $N = 1000$: the `Compact` array is $1000 \times 24 = 24000$ bytes, which
touches $24000 / 64 = 375$ distinct lines. The `Padded` array is
$1000 \times 64 = 64000$ bytes, one line per record, so it touches all 1000.
Both sweeps sum the same values ($0, 1, \dots, 999$), so both sums must agree
(`sum = 499500`) — only the line counts differ:

```
499500.000000 375
499500.000000 1000
ratio=2.666667
```

The padded layout touches 2.67x more cache lines to sweep the exact same
data — the 40 reserved-but-unread bytes per record cost real bandwidth. The
starter never calls `touch()` and returns `0`, so it prints `0.000000 0`
twice and `ratio=0.000000`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, extracts every printed number, and requires `max_abs_err <= 1e-6`
against the same driver linked with `ref.cpp`. Getting the sums right but
forgetting `touch()` (or calling it at the wrong address) leaves the line
counts wrong and fails the gate — the padding cost has to show up in the
printed cache-line counts, not just in the (identical) sums.
