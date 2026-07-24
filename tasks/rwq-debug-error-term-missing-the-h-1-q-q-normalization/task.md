## Context

GPTQ quantizes a weight matrix one column at a time. After quantizing column
$q$, the rounding error it introduced is not just dropped — it is
**propagated forward** into the not-yet-quantized columns, using the inverse
of the layer's (damped) Hessian $H^{-1} = (2 X^\top X + \lambda I)^{-1}$,
computed once from calibration activations $X$. This is what makes GPTQ far
more accurate than naive round-to-nearest: the columns quantized later get to
"absorb" the error created by earlier columns, optimally in a least-squares
sense.

For row block $W \in \mathbb{R}^{r \times d}$ (rows = output features,
columns = input features, quantized in column order $q = 0, 1, \dots, d-1$),
maintaining a working copy $W^{(q)}$ initialized to $W$:

1. Quantize the current column: $\hat{w}_q = \mathrm{Quant}(W^{(q)}_{:,q})$
   using a fixed per-row scale $s$ (given), $\mathrm{Quant}(x) =
   \mathrm{clip}(\mathrm{round}(x/s),\, -L,\, L) \cdot s$ with $L = 2^{\text{bits}-1}-1$.
2. Compute the residual: $e_q = W^{(q)}_{:,q} - \hat{w}_q$.
3. Store the quantized value: $W^{(q+1)}_{:,q} = \hat{w}_q$.
4. **Propagate the error, normalized by the pivot** $[H^{-1}]_{qq}$:
$$
W^{(q+1)}_{:,\,q+1:} \;=\; W^{(q)}_{:,\,q+1:} \;-\; \frac{e_q}{[H^{-1}]_{qq}} \otimes [H^{-1}]_{q,\,q+1:}
$$
(an outer product: the residual column, scaled by the pivot's reciprocal,
times the corresponding row-slice of $H^{-1}$).

The provided `starter.py` implements steps 1–3 correctly but gets step 4
wrong: it propagates the **raw** residual $e_q$ without dividing by
$[H^{-1}]_{qq}$ first. This silently miscalibrates every downstream column's
correction — the sign and scale of $H^{-1}_{qq}$ (which is *not* generally
$1$) matter.

## Task

Fix `gptq_quantize` so the error propagation step divides by the pivot
$[H^{-1}]_{qq}$ as specified above:

```python
def gptq_quantize(W: np.ndarray, Hinv: np.ndarray, scales: np.ndarray, bits: int = 4) -> dict:
    ...
```

* `W` — `(r, d)` float64 weight matrix.
* `Hinv` — `(d, d)` float64 inverse-Hessian matrix (already computed; you do
  not need to invert anything).
* `scales` — `(r,)` float64 array: fixed per-row quantization scale (already
  computed from the original `W`, does not change during the loop).
* `bits` — number of quantization bits (levels are symmetric, $L = 2^{\text{bits}-1}-1$).
* Process columns `q = 0 .. d-1` in order, exactly as specified above.
* Return `{"codes": codes, "W_hat": W_hat}` where `codes` is an `(r, d)`
  integer array (the clipped/rounded code chosen for each element, i.e.
  `round(w_q/s)` clipped to `[-L, L]`, cast to `int`) and `W_hat` is the
  final `(r, d)` float64 dequantized matrix (`W^{(d)}` above).

Vectorised NumPy per column; a single Python loop over the `d` columns is
expected (GPTQ is inherently sequential across columns).

## Example

With `bits=4`, `L=7`. If `scales[i] = 0.01` and `W[i, 0] = 0.023`, the code
for that element is `clip(round(0.023/0.01), -7, 7) = 2`, and the dequantized
value is `2 * 0.01 = 0.02`; the residual `0.003` is then propagated into
columns `1..d-1` of row `i`, scaled by `1/Hinv[0,0]` and `Hinv[0, 1:]`.

## What the gate checks

The grader builds a deterministic weight matrix and a real inverse-Hessian
(from a random calibration-data Gram matrix, damped and inverted with
`np.linalg.inv`), fixed seed, and compares your output to an independent
NumPy oracle running the *correct* algorithm:

* **codes_max_abs_err** — the max absolute difference between your `codes`
  and the oracle's `codes` must be exactly `0` — every element must land on
  the identical integer level as the oracle (the buggy starter's missing
  division changes several downstream codes, so this alone fails it).
* **mse_abs_diff** — `|your MSE(W, W_hat) - oracle's MSE(W, W_hat)|` must be
  at most `1e-6`.
