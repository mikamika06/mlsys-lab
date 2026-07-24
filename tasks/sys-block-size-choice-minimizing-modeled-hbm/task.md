## Context

FlashAttention computes attention by dividing the input matrices into blocks that fit in fast on-chip SRAM. The block sizes determine how often data must be moved from high-bandwidth memory (HBM).

For a sequence length $N$ and head dimension $D$, consider blocks with query size $B_r$ and key/value size $B_c$. A simplified SRAM constraint is

$$
B_rD + B_cD + B_rB_c \le M,
$$

where $M$ is the SRAM element budget.

The modeled HBM traffic counts the reads of key/value blocks and the fixed reads and writes of the query and output matrices:

$$
T(B_r,B_c) =
2ND + 2\left\lceil\frac{N}{B_r}\right\rceil
\left\lceil\frac{N}{B_c}\right\rceil B_cD .
$$

The first term is independent of block choice, while the second term models repeated HBM movement during tiled attention.

## Task

Implement `choose_block_size(M, N, D)`:

```python
def choose_block_size(M: int, N: int, D: int) -> tuple[int, int]:
    ...
```

Return positive integers `(Br, Bc)` satisfying the SRAM constraint and minimizing the modeled HBM traffic $T(B_r,B_c)$.

The grader considers all possible block sizes with $1 \le B_r,B_c \le N` and recomputes the optimum itself. Your returned pair should be within $5\%$ of that minimum modeled traffic.

## Example

```python
M = 256
N = 64
D = 8

Br, Bc = choose_block_size(M, N, D)

# The returned values satisfy:
# Br*D + Bc*D + Br*Bc <= M
# and produce near-minimal modeled HBM traffic.
```

## What the gate checks

The gate exhaustively computes the best feasible block sizes using the traffic equation from the context. It then evaluates the traffic produced by your returned block sizes.

The reported metric `modeled_mem_access` is

$$
\frac{T(B_r,B_c)}{T(B_r^*,B_c^*)},
$$

where $(B_r^*,B_c^*)$ is the oracle minimum. Passing requires this ratio to be at most $1.05$.
