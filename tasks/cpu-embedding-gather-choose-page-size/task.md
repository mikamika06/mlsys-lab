## Context

When gathering rows from an embedding table stored in memory, the access pattern
often appears random. The Translation Lookaside Buffer (TLB) translates virtual
addresses to physical pages. Each TLB entry maps one page, so the number of page
hits and misses depends on both the *stride* of the accesses and the *page size*.

If embedding entries are 64 bytes and a TLB can hold only a small number of
distinct pages, a small page size increases misses, while using *huge pages*
reduces them.

Suppose we have a deterministic cache simulator `cachesim.simulate(addrs,
line_bytes, sets, ways)` that returns a dict with key `'misses'`, counting the
number of unique cache lines touched. We use it as a proxy for page miss
behavior by setting `line_bytes = page_bytes`.

## Task

Implement

```python
def choose_page_size(indices: list[int], row_bytes: int, page_sizes: list[int]) -> int:
    """
    Given the sequence of accessed embedding row indices, each representing
    a contiguous block of row_bytes bytes, simulate address accesses for
    each page size and return the page size (in bytes) that produces the
    fewest simulated TLB misses.
    """
```

Each embedding row has base address `i * row_bytes`. The address trace is the
base address of each accessed index. For each `page_bytes` in `page_sizes`,
simulate the trace by invoking

```python
res = cachesim.simulate(addrs, line_bytes=page_bytes, sets=64, ways=1)
misses = res["misses"]
```

Choose the page size that yields the smallest `misses`. Ties are broken by
smaller page size. Return the *page size* value.

## Example

```python
from arena import cachesim

indices = [1, 2, 8, 1, 4096, 8192, 4096]
page_sizes = [4096, 2**21]
row_bytes = 64

best = choose_page_size(indices, row_bytes, page_sizes)
print(best)  # 2097152 on this deterministic model
```

## What the gate checks

The gate uses the deterministic `cachesim` simulator. It recomputes the misses
for each page size using the same simulator and compares your chosen page size
to the reference's output. Exact match on the chosen value is required for all
fixture patterns; no timing measurement or hand-tuned parameters are involved.
