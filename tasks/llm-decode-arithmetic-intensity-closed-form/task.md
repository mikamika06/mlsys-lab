## Context

The arithmetic intensity (AI) of a computation is defined as the ratio of floating‑point operations to bytes of memory traffic:

$$\mathrm{AI} = \frac{\text{FLOPs}}{\text{bytes}}.$$

In the roofline model, AI determines whether an algorithm is compute‑bound or memory‑bound: low AI indicates that the computation is limited by data movement rather than arithmetic throughput.

For a transformer language‑model during single‑token decoding we can derive closed‑form expressions for both FLOPs and bytes per layer.  
Let

* $L$ – number of layers,
* $H$ – hidden size (dimension of each token embedding),
* $N_h$ – number of attention heads,
* $d_k = H/N_h$ – dimension of a single head,
* $S$ – current sequence length (number of tokens already generated).

### FLOPs per layer

1. **QKV projection**: three linear layers, each $H\times H$, giving $3H^2$ FLOPs.  
2. **Attention matmul**: for each head we compute $QK^\top$ and $QV$. Each requires $d_k \times S$ multiply‑add operations; with $N_h$ heads this is $2\,N_h\,d_k\,S = 2HS$ FLOPs.  
3. **Output projection**: one linear layer $H\times H$, giving $H^2$ FLOPs.  
4. **Feed‑forward network (FFN)**: two linear layers $H\!\to\!4H$ and $4H\!\to\!H$, totaling $8H^2$ FLOPs.

Summing yields

$$
\text{FLOPs}_{\text{layer}} = 12H^{2} + 2HS.
$$

### Bytes per layer

1. **Weights**: all linear layers together contain $12H^2$ scalar weights, i.e. $48H^2$ bytes (4 bytes per float32).  
2. **KV cache**: we read $K$ and $V$, each of shape $(S,d_k)$, so $2\,S\,d_k$ floats or $8SH/N_h$ bytes.  
3. **Input token**: the layer receives a vector of size $H$, costing $4H$ bytes.

Thus

$$
\text{bytes}_{\text{layer}} = 48H^{2} + \frac{8S\,H}{N_h} + 4H.
$$

### Arithmetic intensity

Since both FLOPs and bytes scale linearly with the number of layers, $L$ cancels out:

$$
\mathrm{AI}(L,H,N_h,S) = 
\frac{12H^{2}+2HS}
     {48H^{2} + \dfrac{8S\,H}{N_h} + 4H}.
$$

The function you implement must compute this quantity as a `float64`.

## Task

Implement the following function:

```python
def arithmetic_intensity(num_layers: int,
                         hidden_size: int,
                         num_heads: int,
                         seq_len: int) -> float:
    """
    Return the arithmetic intensity (FLOPs per byte) of a single‑token decode
    for an LLM with the given architecture parameters.
    The result must be a Python float (float64).
    """
```

The implementation should use only pure Python and NumPy; no loops over layers or tokens.

## Example

```python
>>> arithmetic_intensity(12, 768, 12, 128)
0.25632329063683923
```

## What the gate checks

* **Relative error** – The returned AI must match a reference calculation within a relative tolerance of $10^{-9}$.
* **Return type** – The function must return a Python `float` (or NumPy scalar convertible to float).  

Both conditions are enforced by the grader. A correct implementation will pass both gates; any deviation, such as integer division or incorrect formulas, will cause failure.
