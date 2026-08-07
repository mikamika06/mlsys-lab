## Context

A transformer MLP block computes

$$
Y = \mathrm{GELU}(XW_1 + b_1)\,W_2 + b_2,
$$

where $X \in \mathbb{R}^{m \times d}$, $W_1 \in \mathbb{R}^{d \times h}$,
$W_2 \in \mathbb{R}^{h \times d_\text{out}}$.

Megatron-style tensor parallelism shards this block across $N$ ranks without any
communication between the two matmuls:

- **Column-parallel** first layer: $W_1$ is split by columns into
  $W_1 = [W_1^{(0)} \mid W_1^{(1)} \mid \dots \mid W_1^{(N-1)}]$, and $b_1$ is split
  the same way. Rank $i$ computes its own slice of hidden units:
  $$
  A^{(i)} = \mathrm{GELU}\big(X W_1^{(i)} + b_1^{(i)}\big).
  $$
  Because GELU is applied elementwise, splitting the hidden dimension before the
  activation is exactly equivalent to applying GELU to the full hidden vector — no
  synchronization is needed here.

- **Row-parallel** second layer: $W_2$ is split by rows to match, $W_2 =
  \begin{bmatrix} W_2^{(0)} \\ \vdots \\ W_2^{(N-1)} \end{bmatrix}$. Rank $i$ computes
  a partial output $Y^{(i)} = A^{(i)} W_2^{(i)}$. The final result requires a single
  all-reduce sum across ranks, then the (replicated) output bias is added once:
  $$
  Y = \sum_{i=0}^{N-1} Y^{(i)} + b_2 .
  $$

This pattern needs exactly one all-reduce per MLP block instead of one after every
matmul, which is why it is the standard tensor-parallel MLP layout.

Use the tanh approximation of GELU:

$$
\mathrm{GELU}(x) = 0.5\,x\left(1 + \tanh\!\Big(\sqrt{2/\pi}\,\big(x + 0.044715\,x^3\big)\Big)\right).
$$

## Task

Implement `mlp_tensor_parallel(x, w1_shards, b1_shards, w2_shards, b2)`.

Arguments:

- `x`: list, shape $(m, d)$.
- `w1_shards`: list of $N$ list, rank $i$'s array has shape $(d, h_i)$.
- `b1_shards`: list of $N$ list, rank $i$'s array has shape $(h_i,)$.
- `w2_shards`: list of $N$ list, rank $i$'s array has shape $(h_i, d_\text{out})$.
- `b2`: list, shape $(d_\text{out},)$ — the replicated output bias, added once.

The hidden shard sizes $h_i$ may differ between ranks; $\sum_i h_i = h$. Return the
combined MLP output as a `float64` list of shape $(m, d_\text{out})$.

## Example

```python

x = [[1.0, -1.0]]
w1_shards = [[[1.0], [0.5]], [[0.5], [-1.0]]]
b1_shards = [[0.0], [0.0]]
w2_shards = [[[2.0, 0.0]], [[0.0, 2.0]]]
b2 = [0.1, -0.1]

y = mlp_tensor_parallel(x, w1_shards, b1_shards, w2_shards, b2)
```

## What the gate checks

The gate reconstructs the unsharded weights ($W_1$ by concatenating `w1_shards` along
columns, $W_2$ by concatenating `w2_shards` along rows) and evaluates the same
tanh-approximate GELU MLP directly with Python to obtain a reference output. Your
function's output is compared elementwise against this oracle over several random
shard configurations (varying $m$, $d$, per-rank $h_i$, and $d_\text{out}$) using

$$
\text{max\_abs\_err} = \max_{j,k} \left| Y_{jk}^{\text{candidate}} - Y_{jk}^{\text{oracle}} \right| .
$$

The result must satisfy $\text{max\_abs\_err} \le 10^{-5}$. Returning only one rank's
partial second-layer output (skipping the all-reduce), or dropping the GELU
nonlinearity, both fail this gate.
