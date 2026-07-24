## Context

Learnable-rounding post-training quantization (AdaRound and its
descendants) treats "round up or down" as a per-weight decision worth
optimizing, not a fixed rule. Each weight $w$ gets a continuous
**rounding variable** $v \in [-0.5, 0.5]$ that shifts it before rounding:
$w_q = \mathrm{round}(w/s + v)$. When $v$ is near $+0.5$ the weight rounds
up; near $-0.5$ it rounds down; a task loss (e.g. layer output error)
provides a gradient w.r.t. $v$ telling which direction reduces error.
Because $v$ only ever needs to move a rounding boundary, not match an
exact target, production implementations often take **SignSGD** steps on
it — using only the *sign* of the gradient — which is cheap, robust to
gradient-magnitude noise, and naturally bounded, so after each step $v$
is simply clamped back into $[-0.5, 0.5]$.

### One step

Given current rounding variables $V$ (one per weight), a gradient $G$
(one per weight, from the outer objective), and a learning rate
$\eta$ ("lr"):

$$
V' = \mathrm{clip}\big(V - \eta\,\mathrm{sign}(G),\ -0.5,\ 0.5\big)
$$

(using the convention $\mathrm{sign}(0) = 0$, so a zero gradient leaves
that entry of $V$ unchanged this step). Then re-quantize the weights $W$
with fixed scale $s$ using the updated rounding variable:

$$
W_q = \mathrm{clip}\big(\mathrm{round}(W/s + V'),\ q_{min},\ q_{max}\big).
$$

## Task

Implement:

```python
def signsgd_round_step(W, scale, V, grad, lr, qmin, qmax):
    ...
```

* `W` — 1-D array of weights.
* `scale` — fixed positive scalar quantization scale $s$.
* `V` — 1-D array of current rounding variables, one per weight, each in
  $[-0.5, 0.5]$.
* `grad` — 1-D array, gradient of the outer loss w.r.t. `V`.
* `lr` — scalar learning rate $\eta$.
* `qmin`, `qmax` — integer quantization code bounds.

Return `(V_new, W_q)`: the updated rounding variables after one SignSGD
step (clamped to $[-0.5, 0.5]$), and the re-quantized integer codes using
`V_new`, per the formulas above.

## Example

```python
import numpy as np
W = np.array([1.05, -0.42])
scale = 0.5
V = np.array([0.0, 0.0])
grad = np.array([1.3, -0.4])   # positive -> push V down; negative -> push V up
lr = 0.1
V_new, W_q = signsgd_round_step(W, scale, V, grad, lr, qmin=-8, qmax=7)
# V_new = [0.0 - 0.1*sign(1.3), 0.0 - 0.1*sign(-0.4)] = [-0.1, 0.1]
# W/scale = [2.1, -0.84]; W_q = round([2.1-0.1, -0.84+0.1]) = round([2.0, -0.74]) = [2, -1]
```

## What the gate checks

* **max_abs_err** — your `V_new` must match a NumPy oracle applying the
  exact SignSGD-and-clamp formula above, to within $10^{-8}$ absolute
  error, over several random `(W, scale, V, grad, lr)` cases (including
  some exactly-zero gradient entries, to check `sign(0) == 0`).
* **exact_match** — your `W_q` (integer codes) must equal the oracle's
  re-quantization with `V_new` exactly, on the same cases.
