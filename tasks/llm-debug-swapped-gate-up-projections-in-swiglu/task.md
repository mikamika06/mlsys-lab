## Context

The SwiGLU (Switched Gated Linear Unit) is a popular activation used in transformer‑style feed‑forward networks.  
Given an input matrix $X \in \mathbb{R}^{n\times d_{\text{in}}}$ and two linear projections with weight matrices  

$$W_{\text{gate}},\, W_{\text{up}}\;\in\;\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$$

and optional biases $b_{\text{gate}}, b_{\text{up}}\in \mathbb{R}^{d_{\text{out}}}$, the output is  

$$
Y = (X W_{\text{gate}} + b_{\text{gate}})\;\odot\; \sigma(X W_{\text{up}} + b_{\text{up}})
$$

where $\odot$ denotes element‑wise multiplication and $\sigma(z)=\frac{z}{1+e^{-z}}$ is the *silu* (sigmoid‑linear unit).  
The gate projection controls which features are passed through, while the up projection provides a non‑linear scaling.

A common implementation mistake is to apply the silu activation to the wrong branch: using $\sigma(X W_{\text{gate}})$ instead of $\sigma(X W_{\text{up}})$.  This subtle swap changes the semantics of the layer and can lead to degraded performance.

## Task

Implement a function `swiglu` that correctly computes the SwiGLU output as described above. The signature is:

```python
def swiglu(
    X: list[float],
    W_gate: list[float],
    W_up: list[float],
    b_gate: Optional[list[float]] = None,
    b_up: Optional[list[float]] = None
) -> list[float]:
```

* `X` – 2‑D array of shape `(n, d_in)`
* `W_gate`, `W_up` – weight matrices of shape `(d_in, d_out)`
* `b_gate`, `b_up` – optional bias vectors of length `d_out`; if omitted they are treated as zero.

The function must return a list of shape `(n, d_out)` with dtype `float64`.  No explicit Python loops are allowed; use vectorised Python operations only.

## Example

```python

X = [[1.0, 2.0],
              [3.0, 4.0]]
W_gate = [[0.5, -0.2],
                    [0.1, 0.7]]
W_up   = [[-0.3, 0.8],
                   [0.6, -0.5]]

Y = swiglu(X, W_gate, W_up)
print(Y)  # [[0.44789818665375236, -0.10803984064500528], [2.3300872571518845, 0.5268451408989582]]
```

## What the gate checks

The grader computes a reference output using Python and compares it to your implementation with the metric  

$$\max_{i,j} |\,Y_{\text{cand}}(i,j) - Y_{\text{ref}}(i,j)\,|.$$

Your solution must achieve `max_abs_err <= 1e-5` on a set of random test cases.
