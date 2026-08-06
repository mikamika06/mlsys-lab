## Context

A pre-norm transformer block updates a residual stream $x \in \mathbb{R}^{T \times d}$
(a sequence of $T$ tokens, each of width $d$) with two sublayers. The defining
feature of the *pre-norm* wiring is that normalization is applied to the **input**
of each sublayer, and the sublayer output is added **back** onto the un-normalized
stream:

$$
h   = x + \operatorname{attn}\!\big(\operatorname{LN}_1(x)\big),
\qquad
y   = h + \operatorname{mlp}\!\big(\operatorname{LN}_2(h)\big).
$$

Two things make or break this wiring:

1. There are **two** residual adds, and both add onto the stream that entered the
   sublayer ($x$ for the first, $h$ for the second) — never onto the normalized copy.
2. The second normalization sees $h$, the output of the first sublayer, **not** $x$.
   Feeding $x$ again, or moving the norm to the sublayer output (post-norm), changes
   the result.

The pieces are standard. **LayerNorm** over the last axis with population variance
($\mathrm{ddof}=0$) and $\varepsilon = 10^{-5}$:

$$
\operatorname{LN}(z) = \gamma \odot \frac{z - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta,
\quad \mu = \operatorname{mean}(z), \; \sigma^2 = \operatorname{var}(z) .
$$

Single-head scaled dot-product self-attention (no mask):

$$
\operatorname{attn}(z) = \operatorname{softmax}\!\Big(\tfrac{(zW_q)(zW_k)^\top}{\sqrt{d}}\Big)\,(zW_v)\,W_o .
$$

A two-layer MLP with a tanh-approximate GELU activation:

$$
\operatorname{mlp}(z) = \operatorname{gelu}(zW_1 + b_1)\,W_2 + b_2,
\qquad
\operatorname{gelu}(u) = \tfrac{1}{2}u\Big(1 + \tanh\!\big[\sqrt{\tfrac{2}{\pi}}\,(u + 0.044715\,u^3)\big]\Big).
$$

## Task

Implement `pre_norm_block`:

```python
def pre_norm_block(
    x: list[list[float]],
    gamma1: list[float],
    beta1: list[float],
    gamma2: list[float],
    beta2: list[float],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
    Wo: list[list[float]],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
) -> list[list[float]]:
    ...
```

- `x` has shape `(T, d)`.
- `gamma1, beta1, gamma2, beta2` have shape `(d,)` — the two LayerNorms.
- `Wq, Wk, Wv, Wo` have shape `(d, d)` — the attention projections.
- `W1, b1` have shapes `(d, h)`, `(h,)` and `W2, b2` have shapes `(h, d)`, `(d,)`
  — the MLP up- and down-projections.

Return the block output $y$ of shape `(T, d)`, wired exactly as in the Context:
attention sublayer first with `LN1(x)`, residual add, then MLP sublayer with
`LN2(h)`, residual add. Use the LayerNorm ($\varepsilon = 10^{-5}$, population
variance), single-head attention, and tanh-GELU MLP defined above.

## Example

```python
T, d, h = 4, 8, 32
rng = random.Random(1)
x = rng.standard_normal((T, d))
gamma1 = [1.0] * d;  beta1 = [0.0] * d
gamma2 = [1.0] * d;  beta2 = [0.0] * d
Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) / d**0.5 for _ in range(4))
W1 = rng.standard_normal((d, h)) * 0.1; b1 = [0.0] * h
W2 = rng.standard_normal((h, d)) * 0.1; b2 = [0.0] * d

y = pre_norm_block(x, gamma1, beta1, gamma2, beta2,
                   Wq, Wk, Wv, Wo, W1, b1, W2, b2)
assert y.shape == (T, d)
```

## What the gate checks

The grader builds several random blocks (varied `T`, `d`) with a fixed seed,
computes the reference block output with an independent pure-Python implementation,
and measures `max_abs_err` = the maximum absolute difference between your output
and the reference over every element and every case. The gate passes when
`max_abs_err < 1e-5`. Any wiring mistake — re-normalizing $x$ instead of $h$,
dropping a residual add, or using post-norm ordering — pushes the error far above
the threshold.
