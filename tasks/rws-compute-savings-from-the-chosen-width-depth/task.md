## Context

A width/depth pruning method chooses a smaller neural network by removing some
layers and keeping only selected channels in the remaining layers. The parameter
count cannot be computed by multiplying independent width ratios because adjacent
layers are coupled.

For a linear layer with weight matrix $W \in \mathbb{R}^{o \times i}$ and bias
$b \in \mathbb{R}^{o}$, the parameter count is

$$
\mathrm{params}(W,b) = o i + o .
$$

After pruning, if a layer keeps $k_{\mathrm{out}}$ output channels and receives
$k_{\mathrm{in}}$ channels from the previous kept layer, its contribution becomes

$$
k_{\mathrm{out}} k_{\mathrm{in}} + k_{\mathrm{out}} .
$$

Dropped layers contribute zero parameters. The final savings ratio is computed
against the original teacher parameter count:

$$
\mathrm{ratio} =
\frac{\mathrm{teacher\_params}}
{\mathrm{pruned\_params}} .
$$

## Task

Implement `compute_savings_from_chosen_width_depth(layer_shapes, depth_keep, width_keeps)`.

Arguments:

- `layer_shapes` is a list of `(input_width, output_width)` tuples describing
  the teacher's sequential linear layers.
- `depth_keep` is a list of booleans with one entry per layer. `True` means the
  layer remains in the student model.
- `width_keeps` is a list of lists. For each layer, it contains the teacher
  output channel indices kept after pruning. Its length is the chosen output
  width.

Return a tuple:

```python
(pruned_params, ratio)
```

where `pruned_params` is the integer number of parameters remaining and `ratio`
is the teacher parameter count divided by the pruned parameter count.

When computing a kept layer's input width, use the output width of the previous
kept layer. The first kept layer uses the teacher input width from
`layer_shapes[0]`.

## Example

```python
layer_shapes = [(8, 16), (16, 32), (32, 4)]
depth_keep = [True, False, True]
width_keeps = [[0, 1, 2, 3], [], [0, 1]]

compute_savings_from_chosen_width_depth(
    layer_shapes, depth_keep, width_keeps
)
# returns (46, 19.0...)
```

The middle layer is removed. The final layer receives only the four channels
kept by the first layer.

## What the gate checks

The gate builds several pruning configurations and recomputes the reference
answer using an independent parameter-count oracle. The returned parameter count
must exactly match the oracle.

The reported ratio is compared using

$$
1 - \left|r_{\mathrm{candidate}} - r_{\mathrm{oracle}}\right|
$$

and must be at least $0.999999$.
