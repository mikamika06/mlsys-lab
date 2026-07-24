## Context

Arithmetic intensity (AI) is the ratio of floating‑point operations to memory traffic. In a roofline model, AI determines whether a kernel is compute‑bound or memory‑bound. For large language models, decoding involves multi‑head attention (MHA). Grouped‑query attention (GQA) reduces the number of key/value heads that must be read from KV cache, thereby increasing AI.

For an MHA layer during token decoding we approximate:

- **Operations**: Each query vector of dimension $d_k$ is dot‑product with all past keys. For $H$ heads and sequence length $L$, this costs
  $$\text{ops} = H \cdot d_k \cdot L.$$

- **Memory traffic**: We read the current query ($H\,d_k$ bytes) and all past keys/values. If GQA is used, only $H_{\text{kv}}=H/2$ heads provide keys/values; otherwise $H_{\text{kv}}=H$. Thus
  $$\text{mem} = H\,d_k \;+\; 2\,L\,H_{\text{kv}}\,d_k.$$

The arithmetic intensity is $\text{AI}= \frac{\text{ops}}{\text{mem}}$.

## Task

Implement `decode_arithmetic_intensity`:

```python
def decode_arithmetic_intensity(num_heads: int,
                                head_dim: int,
                                seq_len: int,
                                use_gqa: bool) -> float:
    ...
```

The function returns the arithmetic intensity for a single decoding step, using the formulas above. All inputs are positive integers; `use_gqa` indicates whether GQA is enabled.

## Example

```python
>>> decode_arithmetic_intensity(8, 64, 128, False)
0.009765625
>>> decode_arithmetic_intensity(8, 64, 128, True)
0.01953125
```

The second call shows a two‑fold increase in AI when GQA halves the number of KV heads.

## What the gate checks

The grader computes a reference AI with the same formulas and compares it to your output. The relative error must satisfy  
$\displaystyle \frac{|\text{your\_AI} - \text{ref\_AI}|}{\max(|\text{ref\_AI}|,\,10^{-12})}\le 10^{-9}$.
