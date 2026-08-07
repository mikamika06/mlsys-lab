## Context

Mixture-of-Experts (MoE) models route tokens to different experts. During
dispatch, ranks exchange token blocks so that each rank receives the tokens
assigned to its local experts.

For $p$ distributed ranks, let the send buffer be

$$S \in \mathbb{R}^{p \times p \times t \times h},$$

where $S_{i,j}$ is the block of $t$ token representations sent from source rank
$i$ to destination rank $j$. Each token has hidden size $h$.

An all-to-all operation reconstructs the receive buffer for every destination
rank. Destination rank $j$ receives blocks from all source ranks in source order:

$$R_j =
\begin{bmatrix}
S_{0,j} \\
S_{1,j} \\
\vdots \\
S_{p-1,j}
\end{bmatrix}
\in \mathbb{R}^{(pt) \times h}.
$$

This task simulates the communication pattern using list. It does not
perform real distributed communication.

## Task

Implement `moe_all_to_all(send, world_size)`:

```python
def moe_all_to_all(send: list[list[list[list[float]]]], world_size: int) -> list[list[list[float]]]:
    ...
```

The input `send` has shape
`(world_size, world_size, tokens_per_rank, hidden_size)`. The first axis is the
source rank and the second axis is the destination rank.

Return an array with shape
`(world_size, world_size * tokens_per_rank, hidden_size)`. The output at index
`j` must contain the blocks received by destination rank $j$:

$$
\hat{R}_j =
[S_{0,j}; S_{1,j}; \dots; S_{p-1,j}].
$$

The returned values must match the input values exactly up to floating point
precision.

## Example

```python

send = [
        [[[1.0], [2.0]], [[3.0], [4.0]]],
        [[[5.0], [6.0]], [[7.0], [8.0]]],
    ]

out = moe_all_to_all(send, 2)

# out[0]:
# [[1.0], [2.0], [5.0], [6.0]]

# out[1]:
# [[3.0], [4.0], [7.0], [8.0]]
```

## What the gate checks

The gate creates several Python send buffers and computes the expected receive
buffers using the all-to-all definition. It compares the candidate result with
that reference using

$$
\max_i |R_i - \hat{R}_i|.
$$

The reported `max_abs_err` must be less than $10^{-6}$. The cases include
different numbers of ranks, tokens, and hidden dimensions so implementations
that only work for one layout do not pass.
