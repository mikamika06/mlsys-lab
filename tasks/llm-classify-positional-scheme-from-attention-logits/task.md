## Context

Positional encodings inject sequence order information into Transformer models. We can deduce which positional scheme a model uses by inspecting the pre-softmax attention scores $S = Q K^\top$ for a sequence of **identical** tokens (i.e. the input embeddings are the same for all positions). 

For a sequence of identical tokens, let's look at the structure of $S$:
1. **None**: No positional encoding. $S_{i,j}$ is completely constant.
2. **Sinusoidal** (Absolute): $Q_i = (X + P_i)W_Q$, $K_j = (X + P_j)W_K$. The cross terms depend on absolute positions $i$ and $j$ independently, meaning $S_{i,j}$ is **not** a Toeplitz matrix ($S_{i,j} \neq S_{i+1, j+1}$).
3. **RoPE** (Rotary): $Q_i = \mathcal{R}_i X W_Q$, $K_j = \mathcal{R}_j X W_K$. The dot product relies only on the relative distance $i-j$. $S_{i,j}$ is a Toeplitz matrix ($S_{i,j} = S_{i+1, j+1}$), but the values oscillate non-linearly with distance due to the sum of cosines.
4. **ALiBi** (Attention Linear Biases): $S_{i,j} = X W_Q W_K^\top X^\top + m \cdot (j - i)$ (or $m \cdot |i - j|$). This is a Toeplitz matrix where the values change **linearly** as distance increases.

## Task

Write `classify_positional_scheme(S)`:

```python
import numpy as np

def classify_positional_scheme(S: np.ndarray) -> str:
    ...
```

Given a 2D float NumPy array `S` (an attention score matrix for a sequence of identical tokens), return one of `"none"`, `"sinusoidal"`, `"rope"`, or `"alibi"`.

Use a tolerance of `1e-4` for floating-point comparisons.
- `"none"`: All elements in `S` are equal.
- `"sinusoidal"`: `S` is NOT a Toeplitz matrix.
- `"alibi"`: `S` is a Toeplitz matrix AND the change in values along the first row (and first column) is strictly linear (constant step size).
- `"rope"`: `S` is a Toeplitz matrix but NOT linear.

## Example

```python
import numpy as np

# A 4x4 ALiBi score matrix
S = np.array([
    [10.0,  9.0,  8.0,  7.0],
    [ 9.0, 10.0,  9.0,  8.0],
    [ 8.0,  9.0, 10.0,  9.0],
    [ 7.0,  8.0,  9.0, 10.0]
])

classify_positional_scheme(S)
# Returns: "alibi"
```

## What the gate checks

The grader will generate valid $S$ matrices mimicking the math of each scheme and test if your function correctly identifies them via structural properties (Toeplitz, linearity, constancy). It checks your string output directly against the reference generator's label.
