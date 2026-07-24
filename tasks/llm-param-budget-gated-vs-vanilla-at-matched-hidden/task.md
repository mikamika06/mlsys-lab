## Context

Feed‑forward neural networks (FFNs) are a core building block of modern language models.
A vanilla FFN consists of two linear layers with an activation in between:

$$
h = \sigma(W_1 x + b_1), \qquad y = W_2 h + b_2,
$$

where $W_1 \in \mathbb{R}^{d_{\text{in}}\times H}$, $W_2 \in \mathbb{R}^{H\times d_{\text{out}}}$ and $H$ is the hidden size.
The total number of learnable parameters (ignoring biases for simplicity) is

$$
P_{\text{vanilla}} = d_{\text{in}}\cdot H + H\cdot d_{\text{out}} .
$$

Gated variants such as SwiGLU or GeGLU replace the single hidden projection with two parallel projections that are multiplied element‑wise before the final linear layer:

$$
h = (W_{1a}x)\odot(W_{1b}x), \qquad y = W_2 h + b_2,
$$

with $W_{1a},\,W_{1b}\in\mathbb{R}^{d_{\text{in}}\times H}$.
The parameter count becomes

$$
P_{\text{gated}} = 2\,d_{\text{in}}\cdot H + H\cdot d_{\text{out}} .
$$

Thus, for the same hidden size $H$ and identical input/output dimensions,
the gated variant uses roughly $50\%$ more parameters than the vanilla version
(when $d_{\text{in}}\approx d_{\text{out}}$, the ratio is $\frac{3}{2}=1.5$).

## Task

Implement `param_counts(input_dim, output_dim, hidden_size)` that returns a tuple
`(vanilla_params, gated_params)`.  
The function should compute the parameter counts described above **without** counting bias terms.

```python
def param_counts(input_dim: int, output_dim: int, hidden_size: int) -> Tuple[int, int]:
    ...
```

## Example

```python
>>> param_counts(768, 768, 2048)
(3145728, 4718592)

# vanilla_params = 768*2048 + 2048*768 = 3_145_728
# gated_params   = 2*768*2048 + 2048*768 = 4_718_592
```

## What the gate checks

The grader computes the ratio

$$
R = \frac{P_{\text{gated}}}{P_{\text{vanilla}}},
$$

averaged over several test cases.  
Your implementation must produce a ratio $R$ that satisfies $R\le 1.68$.  
A correct implementation yields $R\approx 1.61$, while a typical mistake
(e.g. counting three projections instead of two) gives $R>2$ and fails the gate.
