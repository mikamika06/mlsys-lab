## Context

Package-time optimization systems often remove computation whose inputs are already known. A frozen model may contain weights, scales, and biases that never change after deployment. If a node has only constant inputs, the runtime can replace the computation with its precomputed tensor.

For example, an affine transformation can be represented as

$$y = x \cdot w + b.$$

When $w$ and $b$ are frozen constants and $x$ is also a constant tensor, the compiler can evaluate the expression during packaging:

$$y_{\mathrm{folded}} = x_{\mathrm{const}} \cdot w + b.$$

The runtime graph no longer needs the multiplication and addition nodes because their result is stored as one baked constant.

## Task

Implement `fold_constants(nodes)`:

```python
def fold_constants(nodes):
    ...
```

The input `nodes` is a list of dictionaries in topological order. Each node has a unique `"name"` and an `"op"`.

Supported node formats:

- Constant node:

```python
{"name": "w", "op": "const", "value": numpy_array}
```

- Binary operation node:

```python
{"name": "m", "op": "mul", "inputs": ["a", "b"]}
{"name": "y", "op": "add", "inputs": ["m", "bias"]}
```

- Identity node:

```python
{"name": "out", "op": "identity", "inputs": ["y"]}
```

A node can be folded when all of its inputs are already known constants. Evaluate `mul` using NumPy elementwise multiplication, evaluate `add` using NumPy addition, and evaluate `identity` by forwarding the value.

Return a tuple:

```python
(folded_tensor, folded_node_count)
```

where `folded_tensor` is the value of the last node after constant propagation and `folded_node_count` is the number of non-constant nodes that were replaced by baked constants.

## Example

```python
import numpy as np

nodes = [
    {"name": "w", "op": "const", "value": np.array([2.0, 3.0])},
    {"name": "scale", "op": "const", "value": np.array([4.0, 5.0])},
    {"name": "m", "op": "mul", "inputs": ["w", "scale"]},
    {"name": "bias", "op": "const", "value": np.array([1.0, 1.0])},
    {"name": "out", "op": "add", "inputs": ["m", "bias"]},
]

tensor, count = fold_constants(nodes)

# tensor == array([9.0, 16.0])
# count == 2
```

## What the gate checks

The grader builds several frozen computation graphs and computes the expected folded tensor by evaluating the same graph with a NumPy oracle.

The returned tensor must satisfy

$$\max_i |\hat{x}_i - x_i| \le 10^{-9}.$$

The folded node count must match the number of operation nodes whose inputs become compile-time constants. A solution that returns the correct final tensor but does not perform constant folding accounting will fail the count gate.
