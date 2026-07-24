## Context

Graph optimizers in inference runtimes often replace a sequence of primitive tensor
operations with a single fused operator. Layer normalization is commonly represented
as a subgraph that computes statistics and then applies an affine transform.

For a hidden state vector $x \in \mathbb{R}^{d}$, layer normalization computes

$$
\mathrm{LN}(x)_i = \gamma_i \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i ,
$$

where

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

and

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2 .
$$

A runtime may observe the graph pattern:

`ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add`

and replace these nodes with one `LayerNormalization` node. The fused operator must
produce the same numerical result while preserving the set of nodes removed by the
fusion pass.

## Task

Implement `fuse_layernorm_subgraph(nodes, inputs)`.

`nodes` is a list of dictionaries describing graph nodes. Each dictionary has:

- `name`: unique node name.
- `op`: operation name.
- `inputs`: input node names.

`inputs` contains NumPy arrays and parameters:

- `"x"`: hidden states with shape $(n,d)$.
- `"gamma"`: affine scale with shape $(d,)$.
- `"beta"`: affine bias with shape $(d,)$.

Return a dictionary with:

- `"fused_span"`: a list of node names belonging to the detected LayerNorm
  subgraph.
- `"output"`: the fused LayerNormalization result as a NumPy array with dtype
  `float64`.

If the exact LayerNorm pattern is present, return the fused span and compute the
LayerNorm output. The node span must contain the primitive nodes that are replaced
by the fused operator.

## Example

```python
import numpy as np

nodes = [
    {"name": "mean", "op": "ReduceMean", "inputs": ["x"]},
    {"name": "sub", "op": "Sub", "inputs": ["x", "mean"]},
    {"name": "pow", "op": "Pow", "inputs": ["sub"]},
    {"name": "var", "op": "ReduceMean", "inputs": ["pow"]},
    {"name": "eps", "op": "Add", "inputs": ["var"]},
    {"name": "sqrt", "op": "Sqrt", "inputs": ["eps"]},
    {"name": "div", "op": "Div", "inputs": ["sub", "sqrt"]},
    {"name": "scale", "op": "Mul", "inputs": ["div", "gamma"]},
    {"name": "bias", "op": "Add", "inputs": ["scale", "beta"]}
]

result = fuse_layernorm_subgraph(nodes, inputs)
```

The returned `fused_span` is the primitive node set and `output` is the result of
the equivalent LayerNormalization operation.

## What the gate checks

The gate computes a NumPy oracle for LayerNormalization using the hidden-state
fixtures. It compares the returned output with the oracle using the maximum
absolute error

$$
\max_i |y_i-\hat{y}_i|.
$$

The gate also runs the oracle pattern detector and requires the returned
`fused_span` to exactly match the nodes that would be replaced by the fusion.
Implementations that only compute LayerNorm without detecting the graph pattern
will fail.
