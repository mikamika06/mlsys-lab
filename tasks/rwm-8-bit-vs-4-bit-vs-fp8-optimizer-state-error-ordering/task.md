## Context

Stateful optimizers like Adam keep one or two extra tensors per parameter (the
running mean $m$ and running second moment $v$). For a large model these
optimizer states cost 2-3x more memory than the parameters themselves, so
production training stacks (bitsandbytes, DeepSpeed, Transformer-Engine)
compress them: 8-bit blockwise integers, 4-bit blockwise integers, or 8-bit
floating point (fp8). Each format trades memory for reconstruction error
differently, and the ordering of that error is not always the "obvious" one
(fewer bits is not automatically worse for every distribution, because
floating point adapts its precision to magnitude while integer quantization
does not).

This task asks you to implement all three formats exactly as specified below
and report, for a given optimizer-state vector, the reconstruction MSE and
the storage footprint (in bytes) of each.

### Format A — 8-bit blockwise symmetric integer

Split the (1-D) input into contiguous blocks of `block_size` elements. For
each block $b$:
$$
s_b = \frac{\max_i |x_i^{(b)}|}{127}\quad(\text{use } s_b=1 \text{ if the block is all-zero})
$$
$$
c_i^{(b)} = \mathrm{clip}\big(\mathrm{round}(x_i^{(b)} / s_b),\, -127,\, 127\big) \;\; \text{stored as int8}
$$
Reconstruction: $\hat{x}_i^{(b)} = c_i^{(b)} \cdot s_b$.
Storage: the int8 codes (1 byte/element) plus one float32 scale per block.

### Format B — 4-bit blockwise symmetric integer (nibble-packed)

Same blocking, but the quantized range is $[-7, 7]$ (4 bits, symmetric,
15 levels):
$$
s_b = \frac{\max_i |x_i^{(b)}|}{7}\quad(\text{use } s_b=1 \text{ if the block is all-zero})
$$
$$
c_i^{(b)} = \mathrm{clip}\big(\mathrm{round}(x_i^{(b)} / s_b),\, -7,\, 7\big)
$$
Pack two consecutive codes into one byte: offset each code by `+8` (so it
lands in `[1, 15]`), and pack element `2k` into the low nibble and element
`2k+1` into the high nibble: `byte = low | (high << 4)`. Assume the input
length and `block_size` are both even so no padding is needed.
Reconstruction: unpack the nibbles, subtract 8, multiply by the block's
scale.
Storage: `n/2` packed bytes plus one float32 scale per block.

### Format C — fp8-style floating point (no per-block scale)

4 exponent bits, 3 mantissa bits, exponent bias 6. For each element $x$ with
sign $\sigma=\mathrm{sign}(x)$ and magnitude $m = |x|$:

* Representable magnitude range: $m_{\min} = 2^{-6}$, $m_{\max} = (2 - 2^{-3}) \cdot 2^{8}$.
  Clamp $m$ into $[m_{\min}, m_{\max}]$ before quantizing (values already 0 stay 0).
* Exponent: $e = \mathrm{clip}(\lfloor \log_2 m_{\text{clamped}} \rfloor,\, -6,\, 8)$.
* Fraction: $f = m_{\text{clamped}} / 2^{e}$ (in $[1, 2)$), rounded to 3
  mantissa bits: $f_q = \mathrm{round}(f \cdot 8) / 8$.
* If rounding overflowed to $f_q = 2$ and $e < 8$: renormalize, $f_q \mathrel{/}= 2$, $e \mathrel{+}= 1$.
  Otherwise clip $f_q$ to $2 - 2^{-3}$.
* Reconstruction: $\hat{x} = \sigma \cdot f_q \cdot 2^{e}$ (and $\hat{x}=0$ if $x=0$).

No block scale is stored — each element is self-describing, 1 byte/element.

## Task

Implement:

```python
def optimizer_state_quant_compare(v: np.ndarray, block_size: int = 32) -> dict:
    ...
```

* `v` — 1-D `float64` NumPy array (an optimizer state, e.g. Adam's second
  moment), with `len(v)` and `block_size` both even and `len(v)` divisible by
  `block_size`.
* Returns a `dict` with exactly these keys:
  * `"mse_8bit"`, `"mse_4bit"`, `"mse_fp8"` — reconstruction MSE
    ($\mathrm{mean}((v - \hat v)^2)$) for formats A, B, C respectively.
  * `"bytes_8bit"`, `"bytes_4bit"`, `"bytes_fp8"` — total storage in bytes for
    each format (codes/packed bytes + float32 scales where applicable), as
    plain Python `int`.

Use vectorised NumPy; no explicit Python loops over elements.

## Example

For `v = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.0, 0.25])` and
`block_size = 4`, `optimizer_state_quant_compare(v, 4)` returns a dict whose
`mse_8bit` is far smaller than `mse_4bit` (8 bits resolve the block far more
finely than 4 bits), `bytes_8bit == 4*1 + 2*4 == 12`,
`bytes_4bit == 4*1 + 2*4 == 12` (nibble packing halves the codes but the
scale overhead stays), and `bytes_fp8 == 8`.

## What the gate checks

The grader builds a realistic optimizer-state vector by running a deterministic
Adam-style second-moment EMA over synthetic gradients (fixed seed), calls your
function, and compares against an independent NumPy oracle implementing the
same three formats:

* **rel_err** — relative L2 error between your `[mse_8bit, mse_4bit, mse_fp8]`
  vector and the oracle's (tolerance allows for harmless rounding-convention
  differences, e.g. round-half-to-even vs. round-half-away-from-zero).
* **order_match** — your three MSE values must be ranked in the *same order*
  (smallest to largest) as the oracle's on this data — this is the actual
  "error ordering" the task is about, and it is derived from real numbers,
  not hardcoded.
* **bytes_ok** — your three byte counts must exactly match the oracle's
  (these are deterministic given `len(v)` and `block_size`, independent of
  data).
