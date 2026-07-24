## Context

Mixture-of-Experts (MoE) models route tokens to experts. With expert parallelism (EP), experts are distributed across $N$ workers, so routed token representations must be exchanged before expert computation and exchanged back after computation.

Assume a batch has $T$ tokens, each token is routed to $k$ experts, and each expert receives a vector of dimension $d$. The dispatch phase sends one copy of every routed token representation to its destination expert. The combine phase returns one representation per routed expert output back to the originating worker.

If each element uses $b$ bytes, the communicated volume for one phase is

$$
V_{\mathrm{phase}} = T \times k \times d \times b .
$$

The total all-to-all traffic model counts both dispatch and combine:

$$
V_{\mathrm{total}} = 2 \times T \times k \times d \times b .
$$

This task models bytes moved, not latency or network topology effects.

## Task

Implement `moe_ep_comm_bytes(T, k, d, N, bytes_per_elem=2)`:

```python
def moe_ep_comm_bytes(T: int, k: int, d: int, N: int, bytes_per_elem: int = 2) -> int:
    ...
```

Return the modeled number of bytes moved by dispatch and combine for expert parallelism.

Arguments:

- `T` is the number of routed tokens.
- `k` is the number of experts selected per token.
- `d` is the hidden dimension of each routed representation.
- `N` is the number of expert-parallel workers. It is part of the system model and is guaranteed to be positive, but this simplified volume model does not change with `N`.
- `bytes_per_elem` is the storage size of one tensor element.

The return value must be an integer byte count.

## Example

```python
bytes_moved = moe_ep_comm_bytes(1024, 2, 4096, 8, 2)
# bytes_moved == 33554432
```

## What the gate checks

The gate computes the expected value from a NumPy-based oracle implementation of the communication model and compares the returned byte count exactly.

The `modeled_mem_access` score is `1.0` only when all tested configurations match the oracle value.
