## Context

In transformer‑style attention, the key ($K$) and value ($V$) matrices are often quantized per token (row).  
For a matrix $X \in \mathbb{R}^{n\times d}$ we compute an **absolute‑max scale** for each row

$$s_i = \max_{j} |X_{ij}| ,\qquad i=1,\dots,n.$$

The quantized representation stores the integer values
$\tilde X_{ij}= \operatorname{round}\!\bigl(\tfrac{X_{ij}}{s_i}\bigr)$ as 8‑bit signed integers, together with the per‑row scale $s_i$ stored as a 32‑bit float.

The memory cost of this scheme is

$$
\text{mem}_{\text{quant}}
= n\,d \cdot 1_{\text{byte}} + n \cdot 4_{\text{bytes}}
$$

for one matrix.  
For the pair $(K,V)$ we sum the two costs.

The **size ratio** compares the original floating‑point memory (float32) to the quantized memory:

$$
\text{ratio} = 
\frac{\bigl(\lvert K\rvert + \lvert V\rvert\bigr)\cdot 4}
     {\bigl(n\,d_{\!K}+n\,d_{\!V}\bigr)\cdot 1 + n\cdot 8},
$$

where $d_{\!K}$ and $d_{\!V}$ are the column counts of $K$ and $V$, respectively.

## Task

Implement `compute_scales_and_size`:

```python
def compute_scales_and_size(K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Parameters
    ----------
    K : np.ndarray of shape (n, d_K), dtype float32 or float64
        Key matrix.
    V : np.ndarray of shape (n, d_V), dtype float32 or float64
        Value matrix.

    Returns
    -------
    scales_K : np.ndarray of shape (n,), dtype float64
        Absolute‑max scale per row of K.
    scales_V : np.ndarray of shape (n,), dtype float64
        Absolute‑max scale per row of V.
    size_ratio : float
        Ratio of original (float32) memory to quantized memory as defined above.
    """
```

The function must use only NumPy operations, no explicit Python loops.  
All returned arrays should be `np.float64`.  The scalar may be a plain Python `float` or a zero‑dimensional array.

## Example

```python
import numpy as np
K = np.array([[1., -2.], [3., 0.]])
V = np.array([[-1., 4.], [0., -5.]])
scales_K, scales_V, ratio = compute_scales_and_size(K, V)
print(scales_K)   # [2. 3.]
print(scales_V)   # [4. 5.]
print(ratio)      # 1.3333333333333333
```

The original memory is $32$ bytes (8 float32 values).  
Quantized memory: $8$ bytes for the integers plus $16$ bytes for the scales, total $24$.  
Thus the ratio is $32/24 = 4/3 \approx 1.33$.

## What the gate checks

Two metrics are evaluated:

* **rel_err_scales** – the maximum of the relative errors between the reference and student scale arrays for $K$ and $V$, computed with `arena.scorers.rel_err`.  
  The solution must satisfy $\text{rel\_err} \le 1\times10^{-8}$.

* **size_ratio_err** – the relative error between the reference and student size ratios, also using `arena.scorers.rel_err`.  
  The solution must satisfy $\text{rel\_err} \le 1\times10^{-9}$.

Both metrics are computed on a handful of random test cases.  A correct implementation will pass both gates; any deviation in the scaling or memory accounting will cause a gate failure.
