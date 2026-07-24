## Context

A Translation Lookaside Buffer (TLB) caches recent virtual-to-physical page translations. A TLB with $E$ entries behaves like a small fully associative cache: when a page number is accessed, a hit occurs if the page is resident. On a miss, the page is inserted and the least recently used page is evicted when the TLB is full.

For a byte address $a$ and page size $P$, the accessed page number is

$$
\mathrm{page}(a) = \left\lfloor \frac{a}{P} \right\rfloor .
$$

The number of TLB misses depends on the reuse distance of pages. A loop that repeatedly touches a small working set can keep translations in the TLB, while a large streaming traversal can continuously evict entries.

In this task, the generated trace is evaluated by a deterministic simulator configured as a cache-like device. The simulator models the fixed associativity and replacement behaviour instead of measuring real hardware.

## Task

Implement `make_tlb_trace(pages, rounds, page_size)`:

```python
def make_tlb_trace(pages: int, rounds: int, page_size: int) -> list[int]:
    ...
```

Return a list of byte addresses representing a memory access trace.

The trace should access page-aligned addresses only. The intended pattern is a repeated cyclic walk over the first `pages` pages for `rounds` repetitions:

$$
0, P, 2P, \ldots, (N-1)P
$$

where $P$ is `page_size` and $N$ is `pages`.

The returned list length must be $N \times R$, where $R$ is `rounds`.

## Example

```python
trace = make_tlb_trace(3, 2, 4096)

# [0, 4096, 8192, 0, 4096, 8192]
```

The addresses correspond to page numbers $0, 1, 2, 0, 1, 2$.

## What the gate checks

The gate runs the returned trace through a deterministic simulator configured with fixed parameters. It computes the simulator miss count and compares it with the miss count produced by the reference implementation.

The metric is `exact_match`. The check does not use wall-clock measurements or the machine's real TLB. Only the generated access pattern and the deterministic simulator affect the result.
