## Context

Nearest-rounding (RTN) picks, for every weight, the closer of the two
adjacent quantization levels. But "closer per-weight" is not the same as
"best for the layer's output" — production quantizers like **AdaRound** /
**AutoRound** instead search over the rounding *direction* (down vs. up)
for each weight to directly minimize the quantized layer's output error,
often keeping every weight within one grid step of nearest-rounding but
choosing, jointly, the combination that reconstructs the layer's output
best. This task implements a small, exactly-solvable version of that idea:
**brute-force optimal rounding direction** per output row, compared
against plain nearest-rounding, on the same quantization grid.

### Setup

For weight row $w \in \mathbb{R}^{d_{in}}$ (one output channel) and bit
width $b$, use a symmetric per-row scale:

$$
q_{\max} = 2^{b-1}-1, \qquad s = \frac{\max_j |w_j|}{q_{\max}} \ \ (\text{or } 1 \text{ if } \max_j|w_j|=0)
$$

For each column $j$, the two rounding candidates are
$$
f_j = \mathrm{clip}(\lfloor w_j / s\rfloor,\, -q_{\max},\, q_{\max}), \qquad
c_j = \mathrm{clip}(\lceil w_j / s\rceil,\, -q_{\max},\, q_{\max}).
$$

Given calibration activations $X \in \mathbb{R}^{n_{cal}\times d_{in}}$ and
the true output $y = Xw \in \mathbb{R}^{n_{cal}}$:

* **RTN** (nearest rounding): $\mathrm{code}_j = \mathrm{clip}(\mathrm{round}(w_j/s),-q_{\max},q_{\max})$
  for every $j$; reconstruction $\hat w = \mathrm{code}\cdot s$.
* **Learned rounding**: choose, independently for each row, a binary
  vector $b\in\{0,1\}^{d_{in}}$ (picking $f_j$ or $c_j$ per column) that
  **minimizes** $\lVert X\hat w - y\rVert^2$ over **all** $2^{d_{in}}$
  possible choices — the row-optimal rounding direction.

Because RTN's choice is itself one of the $2^{d_{in}}$ candidates, the
learned-rounding error can never exceed RTN's.

The mean squared output error of a rounding scheme over all $d_{out}$ rows
and $n_{cal}$ calibration samples is
$$
\mathrm{mse} = \frac{1}{d_{out}\cdot n_{cal}}\sum_{i=1}^{d_{out}} \lVert X\hat w_i - Xw_i\rVert^2 .
$$

## Task

Implement `rounding_output_mse`:

```python
def rounding_output_mse(W: list[list[float]], X: list[list[float]], nbits: int) -> tuple[float, float]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix (each row is one output channel).
* `X` — `(n_cal, d_in)` calibration activation matrix.
* `nbits` — bit width $b$ for the symmetric per-row grid above.

Return `(mse_learned, mse_rtn)`:

* `mse_rtn` — mean squared output error using nearest rounding (RTN), as
  defined above.
* `mse_learned` — mean squared output error using, for every row, the
  brute-force-optimal rounding direction over all $2^{d_{in}}$
  combinations, as defined above.

## Example

```python
rng = random.Random(0)
X = rng.normal(size=(30, 6))
W = rng.normal(size=(4, 6))
mse_learned, mse_rtn = rounding_output_mse(W, X, nbits=3)
# mse_learned <= mse_rtn
```

## What the gate checks

* **learned_rel_err** — relative error between your `mse_learned` and a
  Python oracle that brute-forces the optimal rounding direction per row
  (over several random `(W, X)` trials).
* **rtn_rel_err** — relative error between your `mse_rtn` and the oracle's
  nearest-rounding MSE.
* **learned_le_rtn** — your own `mse_learned` must never exceed your own
  `mse_rtn` (learned rounding, done correctly, can only help or tie).
