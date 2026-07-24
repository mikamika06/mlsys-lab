## Context

Production inference servers must decide how many requests can share a fixed KV-cache memory pool. A slot-based allocator reserves the full context window for every request. If each request reserves $n_{\mathrm{ctx}}$ tokens, the memory required for $k$ requests is

$$
M_{\mathrm{slot}}(k) = k \cdot n_{\mathrm{ctx}} \cdot b ,
$$

where $b$ is the memory cost of one token position.

Paged KV-cache allocators reserve memory in blocks. A request with length $L$ uses

$$
\left\lceil \frac{L}{B} \right\rceil
$$

blocks when the block size is $B$. The total memory required for a set of admitted requests is

$$
M_{\mathrm{paged}} = \sum_i \left\lceil \frac{L_i}{B} \right\rceil B b .
$$

The slot allocator has predictable capacity but wastes memory when requests are shorter than the configured context length. A paged allocator can pack variable length requests more efficiently.

## Task

Implement `admission_capacity(memory_budget, n_ctx, block_size, request_lengths)`.

The function receives:

- `memory_budget`: available KV-cache memory in bytes.
- `n_ctx`: maximum context length reserved by a slot-based allocator.
- `block_size`: number of tokens in one paged allocation block.
- `request_lengths`: a list of request token lengths.
- Each token position uses one byte in this simplified model.

Return a tuple:

```python
(slot_capacity, paged_capacity)
```

where:

- `slot_capacity` is the maximum number of requests that fit using contiguous per-slot `n_ctx` allocation.
- `paged_capacity` is the maximum number of requests that fit using the paged allocator.

Requests are considered in the given order. The capacity is the largest prefix of `request_lengths` that fits within the memory budget.

## Example

```python
result = admission_capacity(
    memory_budget=100,
    n_ctx=30,
    block_size=8,
    request_lengths=[7, 12, 20, 4],
)

# slot allocation:
# 3 requests fit because 3 * 30 <= 100
#
# paged allocation:
# 4 requests fit because
# ceil(7/8)*8 + ceil(12/8)*8 + ceil(20/8)*8 + ceil(4/8)*8 = 64
#
# result == (3, 4)
```

## What the gate checks

The gate recomputes the expected capacities using an independent allocator model and compares the returned tuple exactly.

The slot calculation uses

$$
k \cdot n_{\mathrm{ctx}} \leq M
$$

while the paged calculation uses the block rounding rule

$$
\sum_i \left\lceil \frac{L_i}{B} \right\rceil B \leq M .
$$

Only implementations that correctly model both admission strategies pass.
