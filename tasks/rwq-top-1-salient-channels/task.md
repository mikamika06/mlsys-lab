## Context

**AWQ** (Activation-aware Weight Quantization) observes that a linear
layer's quantization error is dominated by a small number of *salient*
input channels — the ones whose activations are systematically large — and
protects the corresponding weight columns instead of treating every column
equally. Salience is measured directly from calibration activations
$X \in \mathbb{R}^{n\times C}$ ($n$ calibration tokens, $C$ input channels):
channel $j$'s salience score is its mean absolute activation across the
calibration set,

$$
s_j = \frac{1}{n}\sum_{i=1}^{n} |X_{ij}| .
$$

The **top-1% salient channels** are the $k = \lceil 0.01\, C \rceil$
channels with the largest $s_j$.

## Task

Implement `top_salient_channels`:

```python
def top_salient_channels(X: list[list[float]], frac: float=0.01) -> list[int]:
    ...
```

* `X` — 2-D `float` array of shape $(n,\,C)$: calibration activations.
* `frac` — fraction of channels to select as salient; $k = \lceil \text{frac}\cdot C\rceil$
  (minimum $1$).

Compute $s_j = \text{mean}_i |X_{ij}|$ for every channel $j$, then return a
1-D integer array containing the indices of the $k$ channels with the
largest $s_j$ (any order — it is graded as a **set**, not a sequence).

## Example

```python
X = [
    [1.0, 0.1, -0.2],
    [-1.0, 0.2, 0.1],
    [0.9, -0.1, 0.0],
]
idx = top_salient_channels(X, frac=0.34)   # ceil(0.34*3) = 2
# column-wise mean(|X|): [0.9667, 0.1333, 0.1]
# -> the 2 most salient channels are {0, 1}
```

## What the gate checks

**exact_match** — the grader loads a fixture calibration batch
(`awq_x.npy`, 600 tokens x 200 channels, with two deliberately amplified
channels standing well above the rest) plus a couple of independently
generated synthetic batches at other channel counts and `frac` values,
computes the same salience scores and top-$k$ selection independently, and
checks that your returned index **set** exactly equals the oracle's set on
every case (order does not matter; a wrong count, an off-by-one in the
`ceil`, ranking by raw mean instead of mean-absolute-value, or picking the
*least* salient channels will all produce a mismatched set).
