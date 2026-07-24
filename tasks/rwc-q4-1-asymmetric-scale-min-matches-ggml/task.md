## Context

GGUF's `Q4_1` format is the classic **asymmetric** 4-bit block quantizer
used by `llama.cpp`/`ggml`. Unlike symmetric schemes that quantize around
zero, Q4_1 stores an explicit per-block **minimum**, so the 16 code levels
$\{0,\dots,15\}$ span exactly $[\min, \max]$ of that block — no wasted
codes on values the block never takes.

For a block of $QK=32$ weights $x_0,\dots,x_{31}$:

$$
\min = \min_j x_j, \qquad \max = \max_j x_j, \qquad
d = \frac{\max - \min}{15}, \qquad
\mathrm{id} = \begin{cases} 1/d & d \neq 0 \\ 0 & d = 0 \end{cases}.
$$

Each weight is quantized using the **full-precision** `id` (not yet
rounded to fp16):

$$
x_j' = (x_j - \min) \cdot \mathrm{id}, \qquad
\mathrm{code}_j = \min\!\big(15,\ \lfloor x_j' + 0.5 \rfloor\big).
$$

Only *after* computing the codes are the block's scale and minimum cast
down to fp16 for storage — this is exactly what `ggml`'s
`quantize_row_q4_1_reference` does in C, and it means the codes are
computed at full precision while $d$ and $\min$ are truncated to fp16
*afterwards*.

## Task

Implement `q4_1_quantize`:

```python
def q4_1_quantize(w):
    ...
```

- `w` — a `float32`/`float64` array of shape $(N_b, 32)$: $N_b$
  independent blocks of 32 weights each.

Return a tuple `(d, m, codes)`:

- `d` — array of shape $(N_b,)$: per-block scale $d = (\max-\min)/15$,
  **rounded to float16 precision** (e.g. via `np.float16(d).astype(np.float64)`
  or equivalent) before being returned.
- `m` — array of shape $(N_b,)$: per-block minimum, likewise rounded to
  float16 precision before being returned.
- `codes` — `uint8` array of shape $(N_b, 32)$: 4-bit codes in
  $\{0,\dots,15\}$, computed with the **full-precision** `id` as shown
  above (i.e. *before* $d$ is rounded to fp16 — rounding $d$ first and
  then quantizing with it will shift codes near block boundaries and is
  wrong).

Dequantization (for reference, not required as output) is
$\hat x_j = \mathrm{code}_j \cdot d + m$.

## Example

A block `[-1.0, 0.0, 1.0, 2.0, ...]` (rest identical) has `min=-1.0`,
`max=2.0`, so `d = 3.0/15 = 0.2`. The value `-1.0` quantizes to code `0`,
`2.0` quantizes to code `15`, and `0.0` quantizes to
`round((0.0 - (-1.0))/0.2) = round(5.0) = 5`.

## What the gate checks

The grader builds several $(N_b, 32)$ weight blocks — including one
constant block where `max == min` (must yield `d=0`, all codes `0`,
not a division by zero) — computes the reference `d, m, codes` with the
exact `ggml` Q4_1 formula above, and compares:

- **codes**, byte-for-byte, must match exactly (`byte_exact_fraction == 1.0`)
  — these are integers derived from a deterministic floor, so any
  correct implementation matches bit-for-bit.
- **d and m**, after your fp16 rounding, must match the oracle's within
  $10^{-6}$ (`max_abs_err`).

Using a symmetric (zero-centered) scheme instead of storing an explicit
min, rounding $d$ to fp16 *before* computing the codes, using
round-half-to-even instead of `floor(x+0.5)`, or forgetting the `d==0`
guard will all be caught.
