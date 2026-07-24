## Context

A **Translation Lookaside Buffer (TLB)** caches virtual-to-physical page translations. Each TLB entry covers one page. A **TLB miss** occurs when accessed memory lies on a page not currently in the TLB — requiring a costly page walk.

Typical **base page** size is 4 KB (x86-64) or 16 KB (Apple Silicon). **Huge pages** (2 MB on x86, 2 MB on Mac) dramatically reduce TLB misses: each entry covers 512× more address space than a 4 KB page, so a working set that needs many 4 KB entries may need only one 2 MB entry.

We model a TLB with $S$ sets, $W$ ways, and a fixed **page size** $P$ bytes. A page-aligned address $A$ maps to set $\lfloor A / P \rfloor \bmod S$, and eviction is LRU within the set. The cache simulator `cachesim.simulate` runs this for a given list of byte addresses and page size, returning `{"misses": m}`.

## Task

Implement `tlb_miss_count(addrs, page_size)` that returns the **number of TLB misses** for the given list of byte addresses and page size, using the specified TLB configuration.

**TLB parameters**: $S = 64$ sets, $W = 4$ ways, LRU replacement. Page size $P$ is provided as argument (between 4 KiB and 2 MiB, always a power of two).

## Example

```python
# Access pages 0, 1, 0, 2 (4 KB each)
addrs = [0, 4096, 4096, 8192]
# With 4 KB pages: first access = miss for pages 0,1,2 → 3 misses
print(tlb_miss_count(addrs, 4096))  # 3

# With 2 MB pages: all addresses in same huge page → 1 miss
print(tlb_miss_count(addrs, 2 * 1024 * 1024))  # 1
```

## What the gate checks

The grader calls your function with a fixed address trace and two page sizes (4 KiB and 2 MiB). It runs both through `cachesim.simulate` with the same reference TLB parameters (64 sets, 4 ways, LRU) and compares the two integer counts to your returned pair. The gate `exact_match` passes only if **both** counts match exactly.
