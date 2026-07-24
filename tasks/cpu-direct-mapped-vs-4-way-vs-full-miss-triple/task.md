## Context

A set-associative cache with `num_sets` sets and `ways` lines per set
holds `num_sets * ways` lines total. A byte address maps to a line
$\mathrm{line} = \lfloor \mathrm{addr} / \mathrm{line\_bytes} \rfloor$, and
that line maps to set $\mathrm{set} = \mathrm{line} \bmod \mathrm{num\_sets}$.
Within a set, an LRU (least-recently-used) policy decides what to evict
when a miss arrives and the set is already full.

Associativity is a trade-off knob at **fixed total capacity**: a
direct-mapped cache (`ways = 1`, many sets) indexes finely but tolerates
zero collisions per set -- two different lines that hash to the same set
will evict each other on every access. A fully-associative cache
(`ways = num_sets * ways`, one set) tolerates any collision, since
everything lives in one giant set, but a real fully-associative cache
this large would be impractical hardware. A 4-way cache sits in between:
each set tolerates up to 4 simultaneously-resident lines before it must
evict.

This matters whenever an access pattern happens to alias: if several
distinct addresses share the same low-order set-index bits (a classic
symptom of power-of-two strides), a low-associativity cache thrashes on
that working set even though the cache has plenty of *total* capacity to
hold it -- these are **conflict misses**, as opposed to the unavoidable
first-touch **compulsory misses**.

## Task

Implement:

```cpp
long lru_cache_misses(const long* addrs, int n, int num_sets, int ways);
void miss_triple(const long* addrs, int n, long* out3);
```

`lru_cache_misses` simulates a set-associative LRU cache (`line =
addr / LINE_BYTES`, `set = line % num_sets`, up to `ways` resident lines
per set, LRU eviction) over `addrs[0..n)` and returns the number of
misses.

`miss_triple` runs the exact same trace through three configurations
that all share the same 16-line total capacity but differ in
associativity, and writes the three results to `out3[0..2]`:

| index | config | num_sets | ways |
|---|---|---|---|
| `out3[0]` | direct-mapped | 16 | 1 |
| `out3[1]` | 4-way | 4 | 4 |
| `out3[2]` | fully-associative | 1 | 16 |

## Example

For `LINE_BYTES = 64` and a trace of just two lines `A` and `B` with
`A % num_sets == B % num_sets` (they collide), accessed as
`A, B, A, B`: a direct-mapped set (`ways = 1`) can hold only one of them
at a time, so every access after the first misses (`4` misses total). A
2-way set could hold both simultaneously, so only the first `A` and first
`B` miss (`2` misses total) -- same trace, same total capacity spent on
that set, fewer misses purely because of associativity.

## What the gate checks

`main.cpp` builds one fixed 60-access trace over 8 distinct lines that
are all congruent mod 16 (so they collide into the *same* direct-mapped
set and the *same* 4-way set by construction): two lines are accessed
repeatedly ("hot"), six others are each touched once per pass ("cold").
It prints the miss triple from all three configurations on that one
trace. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's. Getting the LRU
simulation wrong (or not modeling associativity/eviction at all) changes
some or all three printed numbers; the reference's actual triple is
strictly decreasing (direct-mapped worst, 4-way better, fully-associative
best -- least-recently-used, higher-way caches never do worse for a
fixed access pattern than lower-way ones of the same total capacity), so
a shortcut that returns the same number for all three, or an unweighted
guess, cannot match it.
