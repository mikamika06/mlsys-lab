## Context

A 4-bit quantization scheme is really just a **16-level codebook** plus a
rule for picking which level each value snaps to. Three common choices:

- **NF4** (QLoRA's NormalFloat4): 16 levels placed at the quantiles of a
  *standard normal* distribution — non-uniform, dense near 0 and sparse
  at the tails. Optimal (in an MSE sense) when the data really is
  roughly $\mathcal{N}(0, \sigma^2)$, which pretrained weight tensors
  usually are.
- **FP4**: 16 levels following a floating-point-like exponent/mantissa
  spacing — also non-uniform, but with a different, coarser shape than
  NF4's normal-quantile placement.
- **int4-RTN**: a plain **affine** (uniform, min-to-max) 16-level grid —
  round-to-nearest with no distribution-aware shaping at all.

For NF4 and FP4, a weight vector $w$ is first scaled by its absmax into
$[-1, 1]$, snapped to the fixed codebook $C$ (nearest level), then
rescaled back:

$$
\hat{w} = \operatorname{scale} \cdot \arg\min_{c \in C} \left| \frac{w}{\operatorname{scale}} - c \right|, \qquad \operatorname{scale} = \max_i |w_i|
$$

int4-RTN instead uses a min/max affine map over 16 evenly spaced levels
($\text{lo} = \min(w)$, $\text{hi} = \max(w)$,
$s = (\text{hi}-\text{lo})/15$):

$$
\hat{w} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w - \text{lo}}{s}\right),\, 0,\, 15\right) \cdot s + \text{lo}
$$

Which scheme reconstructs a given weight vector best depends entirely on
its actual distribution — NF4 wins when $w$ genuinely looks Gaussian, but
a shape that violates that assumption (e.g. two far-apart clusters) can
make the "shaped" codebooks waste resolution near zero where NF4 and FP4
concentrate their levels, letting the plain uniform grid win instead.

## Task

Implement `nf4_fp4_int4_best(w)`:

```python
def nf4_fp4_int4_best(w: np.ndarray):
    ...
```

Reconstruct `w` with all three schemes above and compute each one's mean
squared reconstruction error. Return `(errors, best)`:

- `errors`: `np.array([mse_nf4, mse_fp4, mse_int4])`.
- `best`: the name of the scheme with the lowest MSE — `"NF4"`, `"FP4"`,
  or `"INT4"`.

Use these exact 16-value codebooks for NF4 and FP4:

```python
NF4 = [-1.0, -0.696192, -0.525073, -0.394917, -0.284441, -0.184773,
       -0.091050, 0.0, 0.079580, 0.160930, 0.246112, 0.337915,
       0.440710, 0.562617, 0.722956, 1.0]

FP4 = [-1.0, -0.66666667, -0.5, -0.33333333, -0.25, -0.16666667,
       -0.08333333, 0.0, 0.08333333, 0.16666667, 0.25, 0.33333333,
       0.5, 0.66666667, 0.83333333, 1.0]
```

## Example

```python
rng = np.random.default_rng(2025)
w = rng.normal(0.0, 1.0, size=4096)
errors, best = nf4_fp4_int4_best(w)
# errors ~= [0.0138, 0.0153, 0.0252]   (NF4 lowest, as expected for a
#   genuinely normal weight)
# best == "NF4"
```

## What the gate checks

The gate runs several weight arrays from seeded generators: a few with
different (but still roughly Gaussian) scales and sizes, plus one
deliberately **bimodal** array (two well-separated normal clusters) built
specifically so the affine int4 grid, not NF4, comes out ahead — checking
that a solution is genuinely comparing three real computed errors rather
than assuming NF4 always wins.

For every case the reference recomputes all three MSEs and the true
argmin with NumPy. Your `errors` array is compared to it with relative L2
error (`rel_err <= 1e-9`), and your `best` string must match the
oracle's exactly — if it doesn't, the case counts as a total failure
regardless of how close the numeric errors were. A solution that always
reports `"NF4"` as the best scheme (a natural but wrong assumption from
only ever testing on clean Gaussian data) will fail on the bimodal case.
