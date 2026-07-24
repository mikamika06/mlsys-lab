## Context

A transformer decode cache stores key and value vectors for each token and each
attention head. For $T$ tokens, $H$ heads, head dimension $D$, and element size
$E$ bytes, the logical element is

$$
K\!V[t,h,k,d],
$$

where $0 \le t < T$, $0 \le h < H$, $k \in \{0,1\}$ selects key or value, and
$0 \le d < D$.

The physical byte address is determined by the layout. A token-major layout can
make the write of a newly produced token compact, while a head-major layout can
make the decode read for one head compact. The important detail is cache-line
traffic, not only arithmetic contiguity. In this task the modeled cache line is
$64$ bytes, and the tested shapes satisfy

$$
2DE = 64.
$$

So one head's complete key and value record for one token fits exactly in one
cache line. The traffic-optimal record layout is therefore token-major by token,
then head, then key or value, then dimension. Its layout id is `THKD`:

$$
\operatorname{addr}(t,h,k,d)
= \operatorname{base}
+ ((((tH + h)2 + k)D + d)E).
$$

This layout writes a whole token as one compact record and reads one head's
stream without mixing unrelated heads into the same cache line.

## Task

Implement `kv_record_layout_trace`:

```python
def kv_record_layout_trace(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    elem_bytes: int,
    base_addr: int = 0,
) -> dict:
    ...
```

Return a dictionary with exactly these fields:

```python
{
    "layout_id": "THKD",
    "write_addrs": [...],
    "read_addrs": [...],
}
```

`write_addrs` must be the byte-address trace for writing the newest token
$t = T - 1$. Emit one address per element, in this logical order:

```python
for h in range(num_heads):
    for k in range(2):
        for d in range(head_dim):
            ...
```

`read_addrs` must be the byte-address trace for a decode read that streams by
head over all existing tokens. Emit one address per element, in this logical
order:

```python
for h in range(num_heads):
    for t in range(num_tokens):
        for k in range(2):
            for d in range(head_dim):
                ...
```

Use integer byte addresses. Do not call the cache simulator yourself. The grader
will run the deterministic simulator on your returned traces.

## Example

```python
out = kv_record_layout_trace(
    num_tokens=3,
    num_heads=2,
    head_dim=2,
    elem_bytes=1,
    base_addr=1000,
)

out["layout_id"]
# "THKD"

out["write_addrs"]
# [1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015]

out["read_addrs"][:8]
# [1000, 1001, 1002, 1003, 1008, 1009, 1010, 1011]
```

In this toy example the address formula is

$$
1000 + ((((t \cdot 2 + h) \cdot 2 + k) \cdot 2 + d) \cdot 1).
$$

## What the gate checks

The grader computes its own reference by enumerating several candidate layouts,
building their access traces, and running `arena.cachesim.simulate` with pinned
cache parameters. It does not use wall-clock timing or hardware counters.

The gate checks two things. First, `exact_match` is $1$ only if your `layout_id`
matches the simulator-selected reference layout and both modeled byte counts
match exactly. Second, `byte_rel_err` is the relative error of your modeled
write and read traffic compared with the reference traffic. It must be $0$.
