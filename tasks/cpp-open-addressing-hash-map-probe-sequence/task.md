## Context

Open-addressing hash maps store all key-value pairs directly in a
contiguous array of $C$ slots, with no external linked nodes. Each slot
(`Slot { bool occupied; long long key; long long value; }`) is 24 bytes
under the LP64 ABI: `bool`(1) + 7 bytes padding (a `long long` needs 8-byte
alignment) + `long long`(8) + `long long`(8).

Inserting a key $k$:

1. The initial target slot is computed from a fixed 64-bit multiplicative
   hash:
   $$h(k) = (k \cdot 11400714819323198485) \bmod 2^{64}, \qquad i_0 = h(k) \bmod C$$
2. If slot $i$ is already occupied, **linear probing** checks successive
   slots $i_{n+1} = (i_n + 1) \bmod C$ until an empty one is found.

## Task

Implement `insert_probe` in `solve.cpp`:

```cpp
int insert_probe(Slot* table, int C, long long k);
```

Compute `i = hash_key(k) % C` (using `hash_key`, already implemented in
`sol.hpp`), then linearly probe forward (wrapping mod `C`) while
`table[i].occupied` is true. At the first empty slot, set
`occupied = true`, `key = k`, `value = k * 2`, and return that slot index.

The fixed driver in `main.cpp` runs three insertion sequences against fresh
tables of different capacities and prints, for each, the table's byte size
and the sequence of slot indices assigned.

## Example

For `C = 8`, keys `[10, 20, 30, 40, 50]` inserted in order: the assigned
slots are `[2, 4, 6, 0, 3]` (some of these collide with earlier insertions
and have to probe forward to find an empty slot).

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across three `(capacity, keys)`
fixtures, including sequences that collide and must probe forward. The
starter always returns slot `0` without checking occupancy or advancing, so
every insertion after the first key silently overwrites slot `0` instead of
probing — wrong for almost every key in every fixture.
