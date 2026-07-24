## Context

LoRA (Low-Rank Adaptation) adds a trainable low-rank update to a frozen
base weight instead of fine-tuning the full matrix. For an input batch
$x \in \mathbb{R}^{n \times d}$ and a frozen layer whose output is
$\mathrm{base} = xW_0 \in \mathbb{R}^{n \times d}$, a single LoRA adapter
adds

$$
y = \mathrm{base} + \gamma \, (xA)B,
$$

where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$ are
the adapter's low-rank factors ($r \ll d$), and $\gamma$ is a scalar
scaling factor (commonly $\gamma = \alpha / r$ for a configured LoRA
$\alpha$). $A$ and $B$ are never multiplied into a dense $d \times d$
matrix at inference time here — the update is applied as two small
matmuls directly against $x$.

## Task

Implement `lora_delta_forward(x, base, A, B, scale)`:

```python
def lora_delta_forward(x, base, A, B, scale):
    ...
```

- `x`: NumPy array of shape $(n, d)$ — the layer's input.
- `base`: NumPy array of shape $(n, d)$ — the frozen base layer's output
  for that same `x` (i.e. $xW_0$; you are not given $W_0$ itself, only
  its output).
- `A`: NumPy array of shape $(d, r)$.
- `B`: NumPy array of shape $(r, d)$.
- `scale`: a Python float, $\gamma$.

Return `base + scale * (x @ A) @ B` as a `float64` NumPy array of shape
$(n, d)$.

## Example

```python
import numpy as np

x = np.array([[1.0, 2.0]])
base = np.array([[0.5, -0.5]])
A = np.array([[1.0], [0.0]])   # (2, 1)
B = np.array([[0.0, 2.0]])     # (1, 2)
scale = 3.0

y = lora_delta_forward(x, base, A, B, scale)
# x @ A = [[1.0]], (x@A) @ B = [[0.0, 2.0]]
# y == base + 3.0 * [[0.0, 2.0]] == [[0.5, 5.5]]
```

## What the gate checks

The gate loads the committed `x.npy`/`base.npy`/`a.npy`/`b.npy` fixture
and recomputes `base + scale * (x @ A) @ B` in `float64` for several
`scale` values. Your output is compared against this oracle with
`max_abs_err`, threshold $10^{-5}$. Applying the factors in the wrong
order (`(xB)A` or `x @ (A @ B)` with the matrices swapped), forgetting
the base term, or forgetting the scale will not match.
