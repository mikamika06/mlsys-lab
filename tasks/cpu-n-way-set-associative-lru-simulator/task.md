## Context

An N-way set-associative cache is the middle ground between two extremes:
**direct-mapped** ($N=1$, every address has exactly one possible slot) and
**fully associative** ($N = \text{total lines}$, any address can go
anywhere). A byte address splits into a line index
($\lfloor \text{addr} / \text{line\_bytes} \rfloor$) and, from that, a *set*
index ($\text{line} \bmod \text{sets}$); within a set, up to `ways` distinct
lines can be resident at once, and when a miss needs to insert a new line
into a full set, the Least Recently Used one is evicted.

The three parameters interact:

- **Direct-mapped** (`ways=1`): each set holds exactly one line. Two
  addresses that hash to the same set can never coexist, no matter how far
  apart in time they're accessed.
- **Low associativity with colliding addresses**: a handful of live
  addresses hashing into the same set can thrash even when the *cache as a
  whole* has plenty of spare capacity elsewhere.
- **High associativity**: a set can hold several recently used lines even
  if their addresses collide, so short-term reuse survives.

## Task

Implement, in `solve.cpp`:

```cpp
int count_misses(const uint64_t* addrs, int n,
                  int line_bytes, int sets, int ways);
```

Build an `N`-way set-associative LRU cache (`sets` sets of `ways` ways each)
from scratch and replay `addrs[0..n)` through it in order:

- line index = `addrs[i] / line_bytes`
- set index = `line % sets`
- a **hit** (the line is already resident in its set) makes it the most
  recently used line in that set — nothing is evicted.
- a **miss** inserts the line into its set, evicting the set's Least
  Recently Used line first if the set is already full (`ways` lines).

Return the total number of misses over the whole trace.

## Example

The driver (`main.cpp`, fixed) runs three traces against three different
geometries:

- **direct_mapped** (`sets=4, ways=1`) — lines `0,1,2,3` swept once, then
  repeated. No two lines share a set, so nothing is ever evicted:
  `misses=4` (only the first pass; the repeat is all hits).
- **conflict_thrash** (`sets=4, ways=2`) — 3 addresses that all hash to the
  same set, round-robin 4 times. 3 live addresses contending for only 2
  ways means every access past the first lap evicts something another
  access still needs: `misses=12` (every single access misses).
- **fully_assoc_reuse** (`sets=1, ways=8`) — lines `0..9` swept once (10
  distinct lines, only 8 ways, so the two oldest get evicted along the way),
  then lines `8` and `9` (the two most recently touched) repeated. Both are
  still resident: `misses=10` (the two repeats hit).

```
direct_mapped sets=4 ways=1 n=8 misses=4
conflict_thrash sets=4 ways=2 n=12 misses=12
fully_assoc_reuse sets=1 ways=8 n=12 misses=10
```

The starter always returns `0`, which is wrong on all three traces.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Evicting the
Most-Recently-Used line instead of the Least-Recently-Used one still gets
`direct_mapped` right (there's only one line per set to evict, so the policy
never matters there) but disagrees with the reference on both
`conflict_thrash` and `fully_assoc_reuse`, where which line gets evicted
changes whether the later repeats hit or miss.
