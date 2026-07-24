## Context

In a feed‑forward network (FFN) the linear layers are represented by weight matrices whose shapes encode the dimensionality of each stage. A vanilla two‑layer MLP has a first linear layer mapping an input of dimension $d_{\text{in}}$ to a hidden representation of size $h$, followed by a second linear layer that maps this hidden vector to an output of size $d_{\text{out}}$. The corresponding weight shapes are therefore

$$
W_1 \in \mathbb{R}^{\,h\times d_{\text{in}}},\qquad
W_2 \in \mathbb{R}^{\,d_{\text{out}}\times h}.
$$

Gated variants such as SwiGLU and GeGLU modify the first stage to produce both an activation vector and a gate vector. In SwiGLU the single linear layer outputs a concatenation of two $h$‑dimensional vectors, so its weight matrix has shape $(2h\times d_{\text{in}})$. The second linear layer is unchanged:

$$
W_1 \in \mathbb{R}^{\,2h\times d_{\text{in}}},\qquad
W_2 \in \mathbb{R}^{\,d_{\text{out}}\times h}.
$$

In GeGLU the first two linear layers are separate: one produces a gate of size $h$, the other an activation of size $h$. Thus we have three weight matrices:

$$
W_{\text{gate}} \in \mathbb{R}^{\,h\times d_{\text{in}}},\qquad
W_{\text{act}}  \in \mathbb{R}^{\,h\times d_{\text{in}}},\qquad
W_2            \in \mathbb{R}^{\,d_{\text{out}}\times h}.
$$

The task is to infer which variant a network implements solely from the list of weight shapes.

## Task

Implement `classify_ffn_variant`:

```python
def classify_ffn_variant(weight_shapes: list[tuple[int, int]]) -> str:
    ...
```

`weight_shapes` is a list of tuples `(out_dim, in_dim)` describing each linear layer in order. The function must return one of the strings `"vanilla"`, `"swi_glu"` or `"geglu"` according to the rules described above.

The implementation should be pure Python and run in constant time for any reasonable input size.

## Example

```python
# Vanilla MLP
shapes = [(128, 64), (10, 128)]
print(classify_ffn_variant(shapes))   # vanilla

# SwiGLU
shapes = [(256, 64), (10, 128)]      # 2*128 = 256
print(classify_ffn_variant(shapes))   # swi_glu

# GeGLU
shapes = [(128, 64), (128, 64), (10, 128)]
print(classify_ffn_variant(shapes))   # geglu
```

## What the gate checks

The grader compares your output string against a reference implementation that follows the same shape‑based rules. The comparison is case‑sensitive and must match exactly one of the three allowed labels. No other values are accepted.
