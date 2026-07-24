## Context

NF4 ("NormalFloat4") weight-only quantization, as used by QLoRA-style
pipelines, packs each weight into a **4-bit code** (two codes per byte) and
keeps a small **per-block scale** so the codes can be dequantized. If a
tensor of $n$ parameters is split into blocks of `blocksize` values, each
block gets its own scale, stored in some dtype with `scale_bits` bits (e.g.
32 for `float32`, 16 for `float16`, 8 for `int8`).

The total number of stored bytes is the packed 4-bit codes plus the scales:

$$
\text{bytes} = \underbrace{\frac{n}{2}}_{\text{4-bit codes}} \;+\; \underbrace{\frac{n}{\text{blocksize}} \cdot \frac{\text{scale\_bits}}{8}}_{\text{per-block scales}}
$$

Dividing by the number of parameters gives the **effective bits per
parameter** — the number that actually matters when comparing this scheme
against, say, plain 8-bit or 16-bit storage:

$$
\text{bits/param} = \frac{8 \cdot \text{bytes}}{n} = 4 + \frac{\text{scale\_bits}}{\text{blocksize}}
$$

The "+4" is the code itself; the second term is scale overhead amortized
over the block — the whole reason blockwise (rather than per-tensor)
quantization has a real storage cost that shrinks as `blocksize` grows.

## Task

Implement `nf4_storage`:

```python
def nf4_storage(n: int, blocksize: int, scale_dtype: str) -> tuple:
    ...
```

* `n` — number of parameters in the tensor (guaranteed divisible by both 2
  and `blocksize` in the graded cases).
* `blocksize` — number of parameters sharing one scale.
* `scale_dtype` — a NumPy dtype name for the per-block scale, e.g.
  `"float32"`, `"float16"`, `"int8"`, or `"uint8"`. Use
  `np.dtype(scale_dtype).itemsize` to get its size in bytes (so
  `scale_bits = 8 * itemsize`).

Return a 2-tuple `(total_bytes, bits_per_param)`:

* `total_bytes` — an `int`, the packed-codes bytes ($n/2$) plus the
  scale bytes ($\frac{n}{\text{blocksize}} \cdot \text{itemsize}$).
* `bits_per_param` — a `float`, equal to $8 \cdot \text{total\_bytes} / n$.

## Example

```python
nf4_storage(n=1024, blocksize=64, scale_dtype="float32")
# codes:  1024/2            = 512 bytes
# scales: (1024/64) * 4     =  64 bytes
# total_bytes = 576
# bits_per_param = 8*576/1024 = 4.5   (== 4 + 32/64)
# -> (576, 4.5)
```

## What the gate checks

A single **exact_match** gate tries several random `(n, blocksize,
scale_dtype)` combinations — with `n` chosen so it divides evenly by 2 and
by `blocksize` — computes the reference `(total_bytes, bits_per_param)`
directly from the formulas above, and requires your `total_bytes` to match
exactly (integer equality) and your `bits_per_param` to match to floating
point precision. Any mismatch, wrong return shape, or exception fails the
gate.
