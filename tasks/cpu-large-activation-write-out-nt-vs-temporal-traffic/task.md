## Context

A normal ("temporal") store is write-allocate: if the target cache line
isn't resident, the cache controller first performs a **read-for-ownership
(RFO)** -- fetching the whole line from DRAM -- before the CPU can write
even one byte of it, and later the now-dirty line must be **written
back** to DRAM again when it's evicted. That is $2\times$ the line's
size in DRAM traffic, even though every byte of the line was about to
be overwritten anyway.

A **non-temporal (streaming) store** skips all of that: it writes
straight to DRAM through a write-combining buffer, never allocates a
cache line, so there is no RFO and nothing to write back -- only
$1\times$ the line's size.

There's a second, easy-to-miss cost. If you write a large one-time
output (e.g. an activation tensor you won't re-read soon) with
*temporal* stores, every line it touches competes for cache capacity
with whatever else is resident -- including data you're about to reuse.
That's cache **pollution**: it evicts your hot working set, so the next
read of that working set misses and has to be **refetched** from DRAM.
A non-temporal write of the same output causes zero pollution, because
it never enters the cache at all.

## Task

Implement

```cpp
long modeled_dram_traffic(long h_bytes, long a_bytes, bool use_nontemporal);
```

`H` is a `h_bytes` "hot" tensor already resident and read again right
after; `A` is an `a_bytes` write-once activation being flushed out. `H`
occupies byte addresses `[0, h_bytes)`, `A` occupies
`[h_bytes, h_bytes + a_bytes)`. Both are exact multiples of
`LINE_BYTES`. Using the cache model and hooks declared in `sol.hpp`
(`reset_cache`, `touch_byte`, `nontemporal_store`):

1. Call `reset_cache()`.
2. Warm `H`: `touch_byte()` every line once. Each miss charges
   `LINE_BYTES` (a plain read-fill).
3. Write `A`, once, line by line: if `use_nontemporal`, call
   `nontemporal_store()` on every line and charge `LINE_BYTES` per
   line; otherwise call `touch_byte()` on every line (they will all
   miss) and charge `2 * LINE_BYTES` per miss (RFO + eventual
   writeback).
4. Re-touch `H`: `touch_byte()` every line again. Each miss (a
   pollution refetch, only possible if step 3 evicted it) charges
   `LINE_BYTES`; each hit charges 0.

Return the sum of all charges.

## Example

With `h_bytes = 4096` (64 lines) and `a_bytes = 65536` (1024 lines, 16x
the whole 16384-byte modeled cache): writing `A` non-temporally never
touches the cache, so `H` is still fully resident in step 4 -- 0
pollution misses. Writing `A` temporally cycles through every set of
the direct-mapped cache four times over (`1024` lines / `256` sets),
which overwrites every set `H` lives in, so step 4 misses on **all**
64 lines of `H`.

## What the gate checks

`main.cpp` runs both variants on the same `h_bytes`/`a_bytes` and
prints both totals plus their ratio. The reference gets
`temporal_bytes=139264`, `nontemporal_bytes=69632` (ratio `2.0`):
`A`'s own write costs exactly $2\times$ more temporally (RFO +
writeback vs. write-only), **plus** the temporal run pays an extra
`4096` bytes re-fetching all of `H` after it gets evicted -- a cost the
non-temporal run never incurs. Charging only `1x` per line regardless
of store type, or forgetting the step-4 pollution refetch entirely,
changes these numbers and fails the `exact_match` gate; a starter that
returns `0` fails outright.
