## Context

LLM activations are hard to quantize because a handful of channels are
"outliers": for a given token's hidden-state vector $x \in \mathbb{R}^d$, a
few entries are far larger than the rest, so a single per-token scale must
stretch to cover them, crushing every ordinary-sized entry into a couple of
quantization levels. Rotation-based quantizers (QuaRot, SpinQuant) fix this
by multiplying every activation vector by a fixed random **orthogonal**
matrix before quantizing — most naturally a **normalized Hadamard matrix**
$H \in \mathbb{R}^{d\times d}$ with $HH^T = I$. Because $Hx$ mixes every
channel of $x$ into every output channel with equal weight $\pm 1/\sqrt d$,
a single huge coordinate of $x$ gets spread across all $d$ output
coordinates instead of dominating one of them — it cannot disappear (the
vector's $\ell_2$ norm is preserved by orthogonality), but the *peak* is
suppressed relative to the *typical* magnitude.

For one token's activation vector $x$ (a row of the batch), define the
peak-to-typical ratio across its $d$ channels as

$$
r(x) = \frac{\max_j |x_j|}{\operatorname{rms}(x)}, \qquad
\operatorname{rms}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2}.
$$

The Sylvester-Hadamard construction builds $H$ recursively from
$H_1 = [1]$ via $H_{2n} = \begin{bmatrix}H_n & H_n\\ H_n & -H_n\end{bmatrix}$,
normalized as $H \leftarrow H/\sqrt d$ so that $HH^T = I$.

## Task

Implement `outlier_ratio_before_after_rotation`:

```python
def outlier_ratio_before_after_rotation(X: list[list[float]]) -> tuple[list[float], list[float]]:
    ...
```

* `X` — a `float` array of shape $(n,\,d)$: $n$ token activation vectors,
  $d$ a power of two.

1. Build the normalized Sylvester-Hadamard matrix $H$ of size $(d,d)$.
2. Rotate the batch: $X_{\text{rot}} = X H^T$.
3. Compute the per-token ratio $r(x)$ (as defined above, over the channel
   axis) for every row of `X` and every row of `X_rot`.

Return `(ratio_before, ratio_after)`, each a 1-D `float64` array of length
$n$: `ratio_before[i]` is $r$ of `X`'s row $i$, `ratio_after[i]` is $r$ of
`X_rot`'s row $i$.

## Example

```python
# one token, 4 channels, one big outlier coordinate
x = [[0.1, -0.2, 8.0, 0.15]]
before, after = outlier_ratio_before_after_rotation(x)
# before[0] is large (one channel dominates); after[0] is smaller —
# the rotation spread that outlier's energy over all 4 output channels.
```

## What the gate checks

* **rel_err** — the grader loads a fixture batch of activations (`rot_x.npy`,
  512 tokens x 64 channels with a handful of systematic outlier channels,
  the pattern real LLM hidden states show) plus a couple of synthetic
  batches at other power-of-two widths, computes `(ratio_before, ratio_after)`
  independently with a Python oracle, and checks the global relative L2 error
  between your concatenated `(ratio_before, ratio_after)` and the oracle's is
  at most $10^{-6}$.
* **peak_drops** — using *your own* returned arrays, the grader checks that
  the worst (maximum) ratio over all tokens actually goes down after
  rotation: $\max_i \text{ratio\_after}_i < \max_i \text{ratio\_before}_i$.
  A wrong Hadamard construction, an un-normalized $H$ (breaks $HH^T=I$, so
  rotation stops preserving the vector's energy the way this measurement
  relies on), or rotating along the wrong axis will typically make this
  qualitative check fail even when individual numbers look plausible.
