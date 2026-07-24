## Context

Post-training quantization reduces the storage cost of a weight matrix by replacing
floating point values with a smaller representation. GPTQ-style methods reduce the
damage from this approximation by propagating the quantization error from already
processed columns into the columns that are still waiting to be quantized.

Consider a weight matrix $W \in \mathbb{R}^{m \times n}$ and a column interaction
matrix $H \in \mathbb{R}^{n \times n}$. When column $j$ is quantized, the
quantization error is

$$e_j = W_j - \hat{W}_j.$$

A simplified GPTQ update adjusts later columns using the interaction strength:

$$W_k \leftarrow W_k - e_j \frac{H_{jk}}{H_{jj}}, \quad k > j.$$

The update changes future quantization decisions. A common debugging mistake is to
compute $\hat{W}_j$ correctly but forget to subtract the propagated error, causing
later columns to be quantized from stale values.

The quantizer used in this task is symmetric uniform quantization. For a column
$x$, the scale is

$$s = \frac{\max(|x|)}{2^{b-1}-1},$$

and the reconstructed values are

$$\hat{x} = \mathrm{clip}\left(\mathrm{round}(x/s), -(2^{b-1}-1), 2^{b-1}-1\right)s.$$

## Task

Implement `gptq_quantize(W, H, bits=4)`.

The function receives:

- `W`: a 2-D NumPy array of floating point weights with shape $(m,n)$.
- `H`: a symmetric positive diagonal-dominant interaction matrix with shape
  $(n,n)$.
- `bits`: the number of quantization bits.

Return a NumPy array containing the quantized-and-reconstructed weight matrix.

Process columns from left to right. For each column:

1. Quantize the current working column using the symmetric quantizer above.
2. Store the reconstructed column in the output.
3. Compute the quantization error.
4. Propagate that error into every unprocessed column using
   $W_k \leftarrow W_k - e_j H_{jk}/H_{jj}$.

The implementation should update the working copy, not the original input array.

## Example

```python
import numpy as np

W = np.array([
    [1.0, 0.7],
    [-0.5, 1.4],
])
H = np.array([
    [2.0, 0.5],
    [0.5, 3.0],
])

Q = gptq_quantize(W, H, bits=4)
```

The exact numbers depend on the quantization procedure, but the second column is
quantized after receiving the error correction from the first column.

## What the gate checks

The gate recomputes the GPTQ reference algorithm directly with NumPy and compares
the submitted output against that oracle using relative error

$$\mathrm{rel\_err} =
\frac{\lVert Q_{\mathrm{submitted}}-Q_{\mathrm{reference}}\rVert_2}
{\lVert Q_{\mathrm{reference}}\rVert_2 + 10^{-12}}.$$

A solution that only quantizes each column independently will fail because it does
not propagate the quantization error into future columns.
