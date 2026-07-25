## Context

A **Translation Lookaside Buffer (TLB)** caches virtual-to-physical page
translations. Each entry covers exactly one page, so a TLB with $S$ sets
and $W$ ways holds at most $S \times W$ distinct pages at once -- its
**reach** is $S \times W \times P$ bytes, where $P$ is the page size. A
**TLB miss** means the accessed address lies on a page not currently
cached, forcing a page-table walk.

The default page size is 4 KiB on x86-64 (16 KiB on Apple Silicon).
**Huge pages** (2 MiB here) make each entry cover $512\times$ more
address space, so the *same* working set that overflows the TLB's reach
under 4 KiB pages can collapse into a single entry under 2 MiB pages.

The driver in `main.cpp` owns a deterministic set-associative LRU TLB
model: $S = 64$ sets, $W = 4$ ways (256 entries total), reconfigurable
page size. `touch(addr)` records one access; `reset_tlb(page_size)`
clears the TLB and sets its page size; `miss_count()` reports misses
since the last reset. This model -- not any real CPU's TLB -- is the
sole source of every number the gate checks.

## Task

Implement `tlb_miss_pair(base, stride, count, passes, out)`.

Replay the SAME access pattern -- touch `base + i*stride` for
$i \in [0, \text{count})$, repeated `passes` times in the same ascending
order each pass -- twice:

1. against a TLB freshly reset with `page_size = 4096` (4 KiB), writing
   the resulting `miss_count()` into `out[0]`;
2. against a TLB freshly reset with `page_size = 2*1024*1024` (2 MiB),
   writing the resulting `miss_count()` into `out[1]`.

Only the page size differs between the two runs -- the addresses touched
are identical.

## Example

With `count=300, stride=4096, passes=3` the working set spans
$300 \times 4096 = 1{,}228{,}800$ bytes:

- Under 4 KiB pages the TLB's reach is $256 \times 4096 = 1{,}048{,}576$
  bytes -- smaller than the working set -- so pages get evicted before
  the next pass revisits them, and the reference measures **740**
  misses across all three passes (300 compulsory in pass 1, then steady
  thrashing in passes 2-3).
- Under 2 MiB pages the entire 1.2 MB working set lies inside a single
  page ($1{,}228{,}800 < 2{,}097{,}152$), so after the very first touch
  every later access is a hit: **1** miss total.

## What the gate checks

The gate is `exact_match` on the full printed output of two fixed
scenarios. It compiles your `tlb_miss_pair` against the fixed driver,
runs it, and byte-compares stdout against the reference's stdout -- both
`out[0]` (4 KiB) and `out[1]` (2 MiB) must match for every scenario.
Returning zeros, running only one page size, forgetting to
`reset_tlb()` between the two runs (so the second run's TLB isn't
actually reconfigured, or still carries state from the first), or
replaying the addresses in a different order all change the miss counts
and fail the gate.
