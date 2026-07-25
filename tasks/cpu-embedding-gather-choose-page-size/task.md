## Context

When gathering rows from an embedding table stored in memory, the access
pattern often looks scattered. The Translation Lookaside Buffer (TLB)
caches virtual-to-physical page translations; each TLB entry maps exactly
one page, so how many *distinct* pages a trace touches -- and how often
those pages get evicted and re-fetched -- depends on both the stride of
the accesses and the page size.

A small page size (say $4096$ bytes) splits an embedding table into many
pages, so a TLB with only a handful of entries constantly evicts and
re-misses. A *huge page* (e.g. $2^{20}$ or $2^{21}$ bytes) can cover the
entire table in one or two entries, turning most accesses into hits.

This task uses a fixed, deterministic fully-associative LRU TLB model
(16 entries) exposed through three hooks declared in `sol.hpp` and
defined in `main.cpp`:

```cpp
void tlb_reset(long page_bytes);   // empty the TLB, fix the page size
void touch_addr(long byte_addr);   // access one byte address
int  tlb_miss_count();             // misses since the last reset
```

## Task

Implement

```cpp
long choose_page_size(const int* indices, int n, int row_bytes,
                       const long* page_sizes, int p);
```

`indices[0..n)` is the sequence of embedding row indices a gather
accesses, in order; row `r` starts at byte address `r * row_bytes`. For
**each** candidate `page_sizes[k]`: `tlb_reset(page_sizes[k])`, then
`touch_addr(indices[i] * row_bytes)` for every `i` in `[0, n)` in order,
then read `tlb_miss_count()`. Return the page size that produced the
fewest misses. If two or more candidates tie, return the **smaller**
page size.

## Example

Driver trace: 64 scattered indices into a 500-row table, `row_bytes=256`
(whole table spans under 128000 bytes), candidates `{4096, 1<<20, 1<<21}`.
The 4 KiB page splits the table across far more pages than the 16-entry
TLB can hold, so it thrashes. Both huge-page candidates are bigger than
the entire table, so each produces exactly one miss (the very first
access) -- an intentional **tie**. The correct answer is `1048576`
(`1<<20`): same miss count as `1<<21`, but smaller, so the tie-break rule
picks it.

## What the gate checks

`exact_match` on the printed `best_page_size`. Skipping the tie-break
(returning the larger of two equally-good candidates), comparing misses
with `<=` instead of `<` (which silently prefers the *last* candidate
seen instead of the first-smallest), or forgetting to `tlb_reset` between
candidates (letting one candidate's TLB state leak into the next) all
change the printed value.
