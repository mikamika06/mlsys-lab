## Context

Megatron-style tensor (intra-layer) parallelism shards an MLP block across
$P$ devices as a **ColumnParallelLinear** (weight split by output columns,
no communication needed on its own) followed by a **RowParallelLinear**
(weight split by input rows, whose *local* partial products must be
summed across devices to be correct):

$$
Y = \mathrm{RowParallel}\big(\mathrm{GeLU}(\mathrm{ColumnParallel}(X))\big)
$$

Two synchronization primitives make the shapes work out and the gradients
correct, conventionally called $f$ and $g$ (Shoeybi et al., *Megatron-LM*):

$$
f(x) = x \ \text{(forward)}, \qquad \frac{\partial f}{\partial x} = \mathrm{all\_reduce}(\cdot) \ \text{(backward)}
$$
$$
g(x) = \mathrm{all\_reduce}(x) \ \text{(forward)}, \qquad \frac{\partial g}{\partial x} = (\cdot) \ \text{(backward, identity)}
$$

$f$ sits right before every `ColumnParallelLinear` (its job: make sure the
backward gradient w.r.t. a replicated input is correctly summed across
devices). $g$ sits right after every `RowParallelLinear` (its job: sum the
row-parallel layer's local partial outputs into the true replicated
output).

When you stack $L$ of these blocks back to back, block $i$'s output edge
*is* block $i{+}1$'s input edge — so that single physical edge carries
**both** $g_i$ (all-reduce on its forward pass) **and** $f_{i+1}$
(all-reduce on its backward pass).

## Task

Implement `classify_allreduce_edges`:

```python
def classify_allreduce_edges(num_blocks: int):
    ...
```

Consider a chain of `num_blocks` stacked tensor-parallel MLP blocks (each
`ColumnParallelLinear -> GeLU -> RowParallelLinear`). The chain has
`2 * num_blocks + 1` edges, in this order:

```
[in_0, mid_0, out_0(==in_1), mid_1, out_1(==in_2), ..., mid_{L-1}, out_{L-1}]
```

- `in_0` — the very first input to the whole chain.
- `mid_i` — the edge *inside* block `i`, between its column-parallel
  output and its row-parallel input.
- `out_i` — block `i`'s output edge. For `i < num_blocks - 1` this is the
  *same* edge as `in_{i+1}`. For `i == num_blocks - 1` it's the chain's
  final output.

Return a list of `2 * num_blocks + 1` string labels, one per edge, each
one of:

- `"none"` — no communication needed in either direction.
- `"fwd_only"` — all-reduce needed on the forward pass only.
- `"bwd_only"` — all-reduce needed on the backward pass only.
- `"both"` — all-reduce needed on both the forward and the backward pass.

## Example

For `num_blocks = 2` there are 5 edges: `[in_0, mid_0, out_0, mid_1, out_1]`.

```python
classify_allreduce_edges(2)
# ["bwd_only", "none", "both", "none", "fwd_only"]
```

- `in_0`: only `f` touches it (no block precedes it) -> `"bwd_only"`.
- `mid_0`, `mid_1`: within-block edges, neither `f` nor `g` touches them -> `"none"`.
- `out_0`: this edge carries both block 0's `g` (forward all-reduce) and
  block 1's `f` (backward all-reduce) -> `"both"`.
- `out_1`: only `g` touches it (no block follows it) -> `"fwd_only"`.

## What the gate checks

The grader independently derives the expected labels for several chain
lengths (`num_blocks` in `{1, 2, 3, 4, 6}`) directly from the $f$/$g$
placement rule above (an $f$ before every column-parallel layer, a $g$
after every row-parallel layer, then merging operators that land on the
same physical edge), and compares them to your output.

`exact_match` is `1.0` only if your returned label list matches the
expected list exactly, for every `num_blocks` tested (must equal `1.0`).
A solution that gets the boundary blocks right but forgets that interior
block-to-block edges need all-reduce in *both* directions — or one that
puts communication on the within-block edge — will disagree on at least
one label and fail this gate.
