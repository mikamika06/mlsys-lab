## Context

llama.cpp's **GGUF** weight formats trade bits-per-weight for
reconstruction quality via blockwise symmetric quantization: split the
tensor into fixed-size blocks, and store one scale per block alongside
low-bit integer codes. **Q4_0** packs each weight into 4 bits (16x
compression vs fp32); **Q8_0** uses 8 bits (4x compression). Both use the
*same* absmax-scaled, symmetric, round-to-nearest recipe — only the code
width (and therefore the grid resolution) differs. This task computes and
compares their reconstruction error **on the same blocks**, so the
bits-per-weight vs. quality trade-off is directly visible.

### Blockwise symmetric quantization

Each row of $W$ is treated as one block (block size = row length,
$d_{in}$). For a block $x \in \mathbb{R}^{d_{in}}$ and code range
$[-q_{\max}, q_{\max}]$:

$$
d = \frac{\max_j |x_j|}{q_{\max}} \quad (\text{or } 1 \text{ if } \max_j|x_j| = 0)
$$

$$
\text{code}_j = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_j}{d}\right),\, -q_{\max},\, q_{\max}\right),
\qquad \hat x_j = \text{code}_j \cdot d
$$

* **Q4_0**: $q_{\max} = 8$ (signed 4-bit range $[-8, 7]$).
* **Q8_0**: $q_{\max} = 127$ (signed 8-bit range $[-127, 127]$).

## Task

Implement `q4_q8_reconstruction_mse`:

```python
def q4_q8_reconstruction_mse(W: np.ndarray) -> tuple[float, float]:
    ...
```

* `W` — `(n_blocks, block_size)` weight matrix; each row is one
  quantization block.

Return `(mse_q4_0, mse_q8_0)`:

* `mse_q4_0` — mean squared error between `W` and its Q4_0 reconstruction
  ($q_{\max}=8$), averaged over **every** element of `W`.
* `mse_q8_0` — same, for the Q8_0 reconstruction ($q_{\max}=127$).

## Example

```python
import numpy as np
W = np.random.default_rng(0).normal(size=(4, 32))
mse_q4, mse_q8 = q4_q8_reconstruction_mse(W)
# mse_q8 << mse_q4 -- 8-bit codes resolve the block far more finely
```

## What the gate checks

* **q4_rel_err** — relative error between your `mse_q4_0` and a NumPy
  oracle running the Q4_0 recipe above on the fixed weight fixture
  (`gguf_w.npy`).
* **q8_rel_err** — relative error between your `mse_q8_0` and the
  oracle's Q8_0 MSE on the same fixture.
* **q8_beats_q4** — your `mse_q8_0` must be strictly lower than your
  `mse_q4_0` (finer codes, same scale recipe, must reconstruct better).
