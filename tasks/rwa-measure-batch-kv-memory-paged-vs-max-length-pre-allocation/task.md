## Context

Large language model serving systems store key-value (KV) cache entries for active requests. A contiguous allocator reserves memory for the maximum sequence length of every request:

$$M_{\mathrm{contig}} = B \cdot L_{\max},$$

where $B$ is the batch size and $L_{\max}$ is the configured maximum length.

Paged KV cache systems divide memory into fixed-size blocks. A request with length $l_i$ consumes

$$\left\lceil \frac{l_i}{s} \right\rceil s$$

tokens of storage when the block size is $s$. The total paged allocation is

$$M_{\mathrm{paged}} = \sum_i \left\lceil \frac{l_i}{s} \right\rceil s.$$

The memory waste ratio compares the amount reserved by contiguous allocation with the actual paged allocation:

$$R = \frac{M_{\mathrm{contig}}}{M_{\mathrm{paged}}}.$$

A larger ratio means max-length pre-allocation reserves more memory than the paged approach needs.

## Task

Implement `kv_memory_waste_ratio(lengths, max_len, block_size)`:

```python
def kv_memory_waste_ratio(lengths: list[int], max_len: int, block_size: int) -> float:
    ...
```

The function receives the generated token lengths for a batch of requests, the maximum configured sequence length, and the KV cache block size. Return the ratio

$$R = \frac{B \cdot L_{\max}}{\sum_i \lceil l_i / s \rceil s}$$

as a Python `float`.

Assume all inputs are positive integers and `block_size` is positive.

## Example

```python
ratio = kv_memory_waste_ratio([10, 33, 70], 100, 16)

# Contiguous allocation:
# 3 * 100 = 300 tokens
#
# Paged allocation:
# ceil(10/16)*16 + ceil(33/16)*16 + ceil(70/16)*16
# = 16 + 48 + 80 = 144 tokens
#
# ratio = 300 / 144
```

## What the gate checks

The gate computes the expected ratio with an independent oracle using NumPy arithmetic. The returned value must match the oracle with relative error below $10^{-9}$.
