## Context

Multi‑head attention (MHA) is a core component of transformer models. Given an input sequence \(X \in \mathbb{R}^{B\times T\times d_{\text{model}}}\), we project it to queries, keys and values with learned weight matrices
\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
\]
where each \(W_\ast \in \mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}\).  
The hidden dimension is split into \(H\) heads of size \(h=d_{\text{model}}/H\).  For head \(i\) we compute the scaled dot‑product attention
\[
\operatorname{Attn}_i = \operatorname{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{h}}\right)V_i .
\]
The outputs of all heads are concatenated and projected once more with \(W_O\) to obtain the final result
\[
Y = \bigl[\operatorname{Attn}_1,\dots,\operatorname{Attn}_H\bigr] W_O .
\]

## Task

Implement `mha_forward` that performs this forward pass. The function signature is:

```python
def mha_forward(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray,
                Wv: np.ndarray, Wo: np.ndarray) -> np.ndarray:
    ...
```

All inputs are 2‑D or 3‑D NumPy arrays of type `float64`.  
The hidden dimension \(d_{\text{model}}\) is guaranteed to be divisible by the fixed number of heads \(H=4\).  
Return a NumPy array of shape \((B,T,d_{\text{model}})\).

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X  = rng.standard_normal((2,3,8))
Wq = rng.standard_normal((8,8))
Wk = rng.standard_normal((8,8))
Wv = rng.standard_normal((8,8))
Wo = rng.standard_normal((8,8))

Y = mha_forward(X,Wq,Wk,Wv,Wo)
print(Y.shape)  # (2,3,8)
```

## What the gate checks

The grader computes a reference implementation with NumPy and compares your output using the metric
\[
\text{max\_abs\_err} = \max_{i,j,k}\bigl|\,Y_{\text{ref}}[i,j,k]-Y_{\text{cand}}[i,j,k]\,\bigr|.
\]
Your solution must satisfy \(\text{max\_abs\_err}\le 10^{-5}\).  
The function should be fully vectorised; any use of explicit Python loops will cause the metric to exceed the threshold.
