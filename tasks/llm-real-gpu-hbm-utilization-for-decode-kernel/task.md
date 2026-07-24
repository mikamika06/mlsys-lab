## Context

Modern large‑language‑model (LLM) inference on GPUs relies heavily on the high‑bandwidth memory (HBM) subsystem.  
The *arithmetic intensity* of a kernel is defined as

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes transferred}},$$

and the roofline model predicts the achievable performance as

$$P_{\text{achievable}} = \min(P_{\text{peak}},\, B_{\text{HBM}}\times \text{AI}),$$

where $B_{\text{HBM}}$ is the peak memory bandwidth.  
For a single decoder step of a transformer, the dominant operations are:

* Three linear projections (Q, K, V) – each $\mathcal O(b\,s\,h^2)$ FLOPs.
* Self‑attention score matrix multiplication – $\mathcal O(b\,s^2\,h)$ FLOPs.
* Two feed‑forward layers – $2\,\mathcal O(b\,s\,h^2\,f)$ FLOPs, where $f$ is the hidden‑multiplier.

Assuming 32‑bit floats, the total memory traffic can be approximated by

$$
\text{bytes} = 8\,b\,s\,h\,(1+3+2f),
$$

which counts reading the input, the three projection weights, and the two feed‑forward weight matrices (each counted once per token), plus writing all intermediate results.

The arithmetic intensity for a decoder step is therefore

$$
\text{AI} = \frac{6\,b\,s\,h^2 + 2\,b\,s\,f\,h^2 + 2\,b\,s^2\,h}{8\,b\,s\,h\,(1+3+2f)}.
$$

## Task

Implement the function `compute_hbm_utilization` that computes this arithmetic intensity.

```python
def compute_hbm_utilization(batch_size: int,
                            seq_len: int,
                            hidden_dim: int,
                            ff_hidden_mult: int = 4) -> float:
    """
    Return the arithmetic intensity (FLOPs / bytes) of a single decoder step
    for the given batch size, sequence length, hidden dimension and feed‑forward
    multiplier. All inputs are integers; the result is a Python float.
    """
```

The function must use only integer arithmetic for the intermediate counts
and return a `float`.  No external libraries beyond the standard library are required.

## Example

```python
>>> compute_hbm_utilization(1, 10, 768)
112.20833333333333
>>> compute_hbm_utilization(2, 20, 1024, ff_hidden_mult=4)
299.2745098039216
```

(The numbers above are the exact arithmetic intensities for the given parameters.)

## What the gate checks

The grader computes the reference value using the same formula and compares it to your output with a relative error metric:

$$\text{rel_err} = \frac{|\,y_{\text{candidate}} - y_{\text{reference}}\,|}{|\,y_{\text{reference}}\,| + 10^{-12}}.$$

The solution must satisfy $\text{rel_err}\leq 1\times10^{-9}$ for all test cases.
