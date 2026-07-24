## Context

In a transformer‑based language model the input tokens are first mapped to dense vectors by an **embedding matrix** $E \in \mathbb{R}^{V\times d}$, where $V$ is the vocabulary size and $d$ is the hidden dimension.  
The output of the network is projected back to token logits through a linear head with weight matrix $W_{\text{head}}\in\mathbb{R}^{V\times d}$.  

If we **tie** the two matrices, i.e. set $W_{\text{head}} = E^\top$, the model shares parameters between input and output.  
Without tying we have two independent matrices; with tying we keep only one.

The number of trainable scalar parameters is therefore

- **Untied:** $2\,V\,d$  (two separate $V\times d$ matrices)
- **Tied:** $V\,d$      (one shared matrix)

Weight tying thus halves the parameter count, yielding a ratio
$$\frac{\text{untied}}{\text{tied}} = \frac{2\,V\,d}{V\,d} = 2.$$

## Task

Implement `param_savings(vocab_size: int, d_model: int) -> Tuple[int, int]`:

```python
def param_savings(vocab_size: int, d_model: int):
    ...
```

The function must return a tuple `(tied_params, untied_params)` containing the exact integer counts of parameters for the tied and untied configurations described above.

## Example

```python
>>> from your_module import param_savings
>>> param_savings(10_000, 768)
(7_680_000, 15_360_000)
```

The ratio `15_360_000 / 7_680_000` equals `2.0`.

## What the gate checks

The grader computes the reference counts using the same formulas and then evaluates the **size_ratio** metric:

$$\text{size_ratio} = \frac{\text{untied_params}}{\text{tied_params}}.$$

A candidate passes if this ratio is exactly `2.0`.  The function must also return the correct tuple; otherwise the ratio will differ.
