## Context

A multilayer perceptron can be represented as a sequence of layers. A linear layer
with weights $W$ and bias $b$ computes

$$
y = xW + b .
$$

A nonlinear activation such as ReLU applies

$$
\mathrm{ReLU}(z) = \max(0, z)
$$

elementwise.

A fully resident execution keeps all layer parameters in memory and evaluates the
network in order. A streamed execution keeps only the current layer resident:
load one layer, compute its output, release the layer parameters, then continue
with the next layer.

For a deterministic inference workload, streaming is correct when it preserves
the same mathematical computation as the resident execution:

$$
f_{\mathrm{stream}}(x) = f_{\mathrm{resident}}(x).
$$

This task models the forward path of a small MLP. The layer data is provided as
a list of parameter dictionaries. The implementation should process layers one
at a time and return the same output as a resident Python implementation.

## Task

Implement `streamed_mlp_forward(layers, x)`:

```python
def streamed_mlp_forward(layers, x):
    ...
```

`layers` is a list of dictionaries. Each dictionary contains:

- `"w"`: a list of layer weights with shape $(d_{in}, d_{out})$
- `"b"`: a list of biases with shape $(d_{out},)$

Apply each layer in order using

$$
x \leftarrow xW + b
$$

and apply ReLU after every layer except the final layer.

The function must return a list containing the final network output. Do not
modify the input arrays.

A streamed implementation should not concatenate or rebuild all weights into one
large structure. It should consume each layer entry, compute the next activation,
and move on.

## Example

```python

layers = [
    {
        "w": [[1.0, -1.0], [0.5, 2.0]],
        "b": [0.0, 1.0],
    },
    {
        "w": [[2.0], [3.0]],
        "b": [0.5],
    },
]

x = [[1.0, 2.0]]

y = streamed_mlp_forward(layers, x)
# matches the resident execution result
```

## What the gate checks

The grader builds a small MLP and input using Python, computes a full-resident
oracle forward pass, and compares it with the submitted streamed implementation.

The reported metric is

$$
\mathrm{max\_abs\_err} =
\max_i |y_i^{\mathrm{stream}} - y_i^{\mathrm{oracle}}|.
$$

The submission passes when this value is at most $10^{-6}$.
