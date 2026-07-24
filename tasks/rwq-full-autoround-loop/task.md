## Context

Post-training weight quantization maps floating point weights into a smaller integer representation. For a symmetric $b$-bit quantizer, the integer range is

$$
q_{\min} = -2^{b-1}, \qquad q_{\max} = 2^{b-1}-1 .
$$

The scale is computed as

$$
s = \frac{\max(|W|)}{q_{\max}},
$$

and a quantized reconstruction is

$$
W_q = q s .
$$

Round-to-nearest (RTN) uses

$$
q = \operatorname{clip}\left(\left\lfloor \frac{W}{s}+0.5 \right\rfloor,q_{\min},q_{\max}\right).
$$

AutoRound improves the rounding decisions by optimizing a continuous rounding offset. This task uses a deterministic SignSGD version. The offset matrix $r$ starts at zero. Each optimization step computes

$$
q = \operatorname{clip}\left(\left\lfloor \frac{W}{s}+r+0.5 \right\rfloor,q_{\min},q_{\max}\right),
$$

then updates the offset using the straight-through gradient estimate

$$
g = \frac{2s}{N}(W_q-W),
$$

$$
r \leftarrow r-\eta\operatorname{sign}(g),
$$

where $N$ is the number of weights.

The implementation keeps the best reconstruction found during the loop. The best candidate is compared using

$$
\operatorname{MSE}(W,W_q)=\frac{1}{N}\sum_i(W_i-W_{q,i})^2 .
$$

Keeping the best candidate ensures the optimized result is never worse than the initial RTN rounding.

## Task

Implement `autoround_block(W, bits, steps, lr, seed)`:

```python
def autoround_block(W: np.ndarray, bits: int, steps: int, lr: float, seed: int):
    ...
```

Return `(W_q, mse)`.

Requirements:

- `W_q` must have the same shape as `W` and dtype `float64`.
- Use the deterministic SignSGD AutoRound loop described above.
- Track the lowest MSE candidate produced by the loop, including the initial RTN candidate.
- Return the best quantized reconstruction and its MSE.
- Use NumPy operations only.

## Example

```python
import numpy as np

W = np.array([[0.2, -0.8], [1.1, -1.5]], dtype=np.float64)

W_q, mse = autoround_block(W, bits=3, steps=10, lr=0.05, seed=0)

print(W_q.shape)
# (2, 2)

print(mse >= 0)
# True
```

## What the gate checks

The gate runs an independent NumPy oracle implementing the same optimization loop. The returned reconstruction must exactly match the oracle result, and the reported MSE must match within $10^{-6}$.

The gate also computes the RTN reconstruction error and verifies that the returned AutoRound result is no worse than RTN. A solution that only returns RTN without running the optimization loop will fail.
