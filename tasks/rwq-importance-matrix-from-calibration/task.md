## Context

Tools like `llama.cpp`'s `imatrix` collect an **importance matrix** by running
a calibration corpus through the model and, for every linear layer's input,
accumulating a per-channel weight from the activations it observes. That
weight is later used to bias GPTQ/RTN-style quantizers toward preserving the
input channels the model actually relies on, instead of treating every
weight column as equally important.

The core statistic is simple. Given a calibration activation matrix
$X \in \mathbb{R}^{n \times d}$ — $n$ calibration tokens by $d$ input
channels — the importance of channel $j$ is the sum, over every token, of
that channel's squared activation:

$$
\text{imatrix}_j = \sum_{i=1}^{n} X_{ij}^2 .
$$

A channel that repeatedly fires with large magnitude across the calibration
set accumulates a large importance weight; a channel that stays near zero
contributes almost nothing.

## Task

Implement `imatrix_from_calibration`:

```python
def imatrix_from_calibration(X: list[list[float]]) -> list[float]:
    ...
```

* `X` — list of lists of floats of shape $(n,\;d)$: $n$ calibration tokens (rows),
  $d$ input channels (columns).

Return a list of floats of length $d$ where entry $j$ is
$\sum_{i=1}^n X_{ij}^2$ — the sum of squares down each column (i.e. summed
over the token axis, not the channel axis). Use vectorised Python; do not
loop over channels or tokens in Python.

## Example

```python
X = [
    [1.0, 0.0, 2.0],
    [2.0, 0.0, -1.0],
    [0.0, 0.0, 3.0],
]
imp = imatrix_from_calibration(X)
print(imp)   # -> [5.0, 0.0, 14.0]
# channel 0: 1^2 + 2^2 + 0^2 = 5
# channel 1: never fires        -> 0
# channel 2: 2^2 + (-1)^2 + 3^2 = 14
```

## What the gate checks

A single gate, **rel_err**, compares your output against a Python oracle that
computes `[sum(x**2 for x in col) for col in zip(*X)]` in float64. Your function is graded on the
fixture calibration matrix `gguf_x.npy` (activations across 800 calibration
tokens and 96 channels, with a few deliberately "hot" high-magnitude
channels) and on an independently generated random calibration batch, so a
solution that only matches the fixture's shape or scale by coincidence will
not pass. The global relative L2 error between your result and the oracle's
must satisfy

$$
\frac{\lVert \hat{v} - v \rVert_2}{\lVert v \rVert_2} \le 10^{-8}.
$$
