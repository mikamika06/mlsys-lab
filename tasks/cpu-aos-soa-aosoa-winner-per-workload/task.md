## Context

Three ways to store $N$ records of $F$ fields each:

- **AoS** (array of structs): record $i$'s fields are contiguous; consecutive records are contiguous. Reading every field of every record is naturally sequential — but reading just *one* field means skipping over the other fields every single record, which wastes whatever cache-line bytes those other fields occupy.
- **SoA** (struct of arrays): each field gets its own contiguous array of $N$ elements. Reading one field is a single tight sequential scan touching nothing else. Reading every field means touching $F$ separate contiguous regions.
- **AoSoA**: a middle ground — records are grouped into fixed-size blocks (here, 8 records per block), and *within* each block the layout is field-major (mini-SoA). One field's data within a block is contiguous, but only for that block's 8 records.

For this exercise, records are packed with **no alignment padding** (byte math stays simple): 4 fields sized `{4, 4, 4, 8}` bytes, `record_size = 20`.

## Task

Implement `void emit_access(Layout layout, int n, int field_idx)`. For every byte address that reading `field_idx` (or, when `field_idx == -1`, every field) of every one of the `n` records would touch **under the given layout's physical arrangement**, call `touch(addr)` once. Where those bytes physically live is entirely determined by `layout`:

- `Layout::AoS`: record `i`, field `f` lives at `i * 20 + offset(f)`.
- `Layout::SoA`: field `f`'s whole array lives at a fixed base (fields laid out back-to-back); record `i` of field `f` is at `base(f) + i * size(f)`.
- `Layout::AoSoA`: record `i` belongs to block `i / 8`, position `i % 8` within it; block `b` starts at `b * 160`, and within it field `f`'s 8-element sub-array starts at a fixed sub-offset.

## Example

For `n = 64` records and `field_idx = -1` (every field), every layout ends up touching the same total `1280` bytes — there's nothing to skip. For `field_idx = 0` (just the 4-byte `id` field), AoS still touches close to all `1280` bytes (every record's 20-byte stride keeps landing in a fresh 64-byte line even though only 4 of those bytes are wanted), while SoA touches only the `256` contiguous bytes that actually hold `id` values.

## What the gate checks

`main.cpp` runs a real deterministic cache-line model (every `touch(addr)` call buckets `addr / 64` into a set; the reported cost is `distinct_lines * 64`) over two workloads — "every field" and "field 0 only" — for all three layouts, and prints each layout's cost plus the winner (lowest cost) per workload. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Getting any layout's address arithmetic wrong changes its touched-line count, and for the field-only workload the AoS/SoA/AoSoA costs are all different by construction, so a wrong formula for any one of them changes both a printed number and, in most cases, the printed winner.
