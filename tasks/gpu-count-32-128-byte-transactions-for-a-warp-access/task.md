## Context

On NVIDIA GPUs, global memory is served through the L2 cache in 128-byte **cache lines** (also called segments). When a warp of 32 threads issues loads or stores, the memory controller inspects every address and groups those that fall within the same aligned 128-byte region into a single **transaction**. The number of transactions determines throughput: one transaction is ideal; many transactions waste bandwidth.

A 128-byte segment is aligned to a 128-byte boundary. Given a byte address $a$, the segment index it belongs to is:

$$s(a) = \left\lfloor \frac{a}{128} \right\rfloor$$

If thread $i$ accesses byte address $a_i$, the total number of transactions is the count of **distinct** segment indices among all active threads.

## Task

Implement the function:

```python
def count_transactions(base_addr: int, stride: int, num_threads: int = 32) -> int:
```

Thread $i$ (for $i = 0, 1, \ldots, \text{num\_threads} - 1$) accesses byte address:

$$a_i = \text{base\_addr} + i \times \text{stride}$$

Return the number of distinct 128-byte segments touched:

$$T = \left|\left\{ \left\lfloor \frac{a_i}{128} \right\rfloor \;:\; i = 0, \ldots, \text{num\_threads} - 1 \right\}\right|$$

Both `base_addr` and `stride` are measured in bytes.

## Example

Coalesced `float32` access: `base_addr=0, stride=4, num_threads=32`. Addresses are $0, 4, 8, \ldots, 124$. All fall in segment $\lfloor 0/128 \rfloor = 0$. Result: **1 transaction**.

Stride-two `float32` access: `base_addr=0, stride=8, num_threads=32`. Addresses are $0, 8, 16, \ldots, 248$. Addresses $0$–$120$ fall in segment $0$; addresses $128$–$248$ fall in segment $1$. Result: **2 transactions**.

Offset coalesced access: `base_addr=64, stride=4, num_threads=32`. Addresses are $64, 68, \ldots, 188$. The first 16 threads ($64$–$124$) are in segment $0$; the remaining 16 ($128$–$188$) are in segment $1$. Result: **2 transactions**.

## What the gate checks

The grader calls your function on a variety of `(base_addr, stride, num_threads)` triples and compares each result against a brute-force reference that enumerates all thread addresses and counts distinct $\lfloor a / 128 \rfloor$ values. Every test case must match exactly.
