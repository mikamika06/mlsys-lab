---
title: "What is the int8 range?"
description: "int8 range explained for int4/int8/int16, symmetric vs asymmetric, with a measured quantization-error table you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is the int8 range?

int8 range is the set of 256 integers a signed 8-bit value can hold: -128 to 127 in the
symmetric scheme used for weights, or 0 to 255 unsigned when a zero-point shifts the scheme to
asymmetric. Halving the bit width to int4 does not halve the error it causes — on a fixed
normal tensor below, dropping from int8 to int4 raises mean absolute quantization error from
0.009 to 0.169, an 18x jump for 4 fewer bits. The measurement that produces that number, and
the full int4/int8/int16 range table, follow.

## How it works

An integer type of `b` bits can represent exactly `2^b` distinct values, full stop — that count
is fixed by the bit width and nothing else. What is *not* fixed is which real numbers those
integers stand for, and that is the entire content of a quantization scheme. **Symmetric**
quantization is signed and centered on zero: the range splits as `-2^(b-1)` to `2^(b-1) - 1`
(int8: -128 to 127, one extra negative slot because two's complement is asymmetric even when
the scheme is not), and a single scale factor maps the tensor's absolute maximum onto the
positive edge. **Asymmetric** quantization is unsigned, `0` to `2^b - 1`, and adds a zero-point
offset so the tensor's true minimum and maximum both land exactly on the integer range's edges,
wasting no codes on values that never occur.

That difference in floor placement is also the whole reason to have two schemes rather than
one. A ReLU activation tensor is never negative — symmetric quantization would spend half its
codes, 128 of 256 for int8, representing negative numbers that cannot appear. Weight tensors,
by contrast, are usually close to zero-centered, so symmetric wastes almost nothing and gets a
cheaper dequantize: `q * scale` instead of `(q - zero_point) * scale`. Picking the wrong scheme
for the wrong tensor is a real, measurable error cost, not a style choice.

Bit width and error do not trade off linearly, which is the part a one-line "int8 = 256 levels"
definition hides. Going from 16 to 8 bits divides the level count by 256 but the levels were
never spread evenly over where the data actually lives — a normal distribution's mass sits
within about 3 standard deviations, so most of a wide integer range is already spent on rare
tail values before the bit width shrinks at all. This is the same "count what the hardware
actually gives you, don't estimate it" instinct as [memory coalescing](memory-coalescing.md),
where a stride that looks harmless costs a countable multiple of memory transactions, and as
[false sharing](false-sharing.md), where a layout that looks fine costs a countable number of
cache-line invalidations. Range and error here are exactly that kind of number: cheap to state
wrong, cheap to check.

The practical range table below covers int4 (used group-wise for weight-only LLM inference),
int8 (the default for both weights and KV-cache), and int16 (rare on its own, but the reference
point for "how much do 8 fewer bits actually cost").

## Ranges and measured quantization error by bit width

Bits and scheme were varied; `min`/`max`/`levels` come directly from the bit width and are not
measured, only the two error columns are. Both are the quantize-dequantize round-trip error on
one fixed seeded `N(0,1)` tensor of 100,000 values (`numpy.random.default_rng(0)`), using
per-tensor scale (and, for asymmetric, zero-point) fit to that tensor's own min/max.

| bits | scheme | min | max | levels | max abs err | mean abs err |
|---|---|---|---|---|---|---|
| 4 | symmetric | -8 | **7** | 16 | 0.337997 | 0.168924 |
| 4 | asymmetric | 0 | 15 | 16 | 0.307534 | 0.153595 |
| 8 | symmetric | -128 | 127 | 256 | 0.018630 | **0.009335** |
| 8 | asymmetric | 0 | 255 | 256 | 0.018090 | 0.009019 |
| 16 | symmetric | -32768 | 32767 | 65536 | 0.000072 | 0.000036 |
| 16 | asymmetric | 0 | 65535 | 65536 | 0.000070 | 0.000035 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=100_000)

def sym_range(bits):
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1, 2 ** bits

def asym_range(bits):
    return 0, 2 ** bits - 1, 2 ** bits

def sym_error(x, bits):
    lo, hi, _ = sym_range(bits)
    scale = np.abs(x).max() / hi
    q = np.clip(np.round(x / scale), lo, hi)
    err = np.abs(x - q * scale)
    return err.max(), err.mean()

def asym_error(x, bits):
    lo, hi, n = asym_range(bits)
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin) / (n - 1)
    zp = np.round(-xmin / scale)
    q = np.clip(np.round(x / scale + zp), lo, hi)
    err = np.abs(x - (q - zp) * scale)
    return err.max(), err.mean()

for bits in (4, 8, 16):
    lo, hi, n = sym_range(bits)
    me, ma = sym_error(x, bits)
    print(f"bits={bits} sym  min={lo} max={hi} levels={n} max_abs_err={me:.6f} mean_abs_err={ma:.6f}")
    lo, hi, n = asym_range(bits)
    me, ma = asym_error(x, bits)
    print(f"bits={bits} asym min={lo} max={hi} levels={n} max_abs_err={me:.6f} mean_abs_err={ma:.6f}")
PY
```

Two things worth more than the row-by-row numbers. First, the 4-to-8-to-16 progression is not
linear: doubling the bits from 4 to 8 cuts mean error by about 18x, but doubling again from 8
to 16 cuts it by another ~260x — each extra bit buys more precision than the last, because it
halves the *step size* while the tensor's spread stays fixed. Second, asymmetric beats
symmetric at every width here even though the input is a zero-centered normal: this particular
100,000-sample draw has `min=-4.494` against `max=4.732`, so symmetric quantization sizes its
whole range off the larger of the two magnitudes and wastes about 5% of its span on negative
codes below `-4.494` that the tensor never uses. That gap would close on a perfectly symmetric
population and would widen sharply on a real skewed activation tensor — which is exactly why
production quantizers pick the scheme per-tensor rather than fixing one globally.

## Practise it

```bash
mlsys grade rwq-symmetric-vs-asymmetric-int8-quantize-dequantize
```

[That task](../tasks/rwq-symmetric-vs-asymmetric-int8-quantize-dequantize/task.md) asks you to
implement exactly the two round-trips measured above and gates on three numbers at once:
`max_abs_err <= 1e-06`, `sym_mse_diff <= 1e-09`, and `asym_mse_diff <= 1e-09` against a NumPy
oracle. The shipped starter is two bare `raise NotImplementedError` stubs, so it fails all
three immediately; the trap once you do implement it is rounding before clipping in the wrong
order, or reusing the symmetric scale formula's `/127` for the asymmetric branch, which passes
`max_abs_err` on easy inputs and still fails `asym_mse_diff` on skewed ones.

More range-and-packing tasks, roughly in order of difficulty:
[round-half-to-even fixed-point quantizer](../tasks/num-round-half-to-even-fixed-point-quantizer/task.md)
(the rounding rule underneath every table row above),
[int8 quantize-dequantize round trip](../tasks/cpp-int8-quantize-dequantize-round-trip/task.md)
in C++,
[per-axis symmetric int8](../tasks/rwq-qint8-per-axis-symmetric-quant/task.md) (one scale per
output channel instead of one per tensor),
[bit-field-packed int4/int8 quant record](../tasks/cpp-bit-field-packed-int4-int8-quant-record/task.md)
(two int4 codes per byte, plus the scale, in one struct), and
[group-wise int4 symmetric weight quant](../tasks/rwc-int4-groupwise-group-32-symmetric-weight-quant/task.md)
(one scale per 32-element chunk, the scheme this page's int4 row simplifies away).

## Common mistakes

- **Treating `2^b` levels as `2^b` usable levels.** Signed two's complement has one more
  negative value than positive (`-128` to `127`, not `-127` to `127`), and code that hardcodes
  a symmetric-looking `[-127, 127]` clip throws away a representable level for nothing.
- **Using `abs(x).max()` for asymmetric scale.** Asymmetric scale must come from `min` and
  `max` separately; reusing the symmetric absmax formula silently degrades to a worse-than-symmetric
  result because it ignores the zero-point that asymmetric quantization exists to add.
- **Assuming halving bit width halves error.** The table shows 8-to-4 bits costs about 18x the
  mean error, not 2x, because error scales with step size, which is exponential in bit width,
  not linear.
- **Picking symmetric for a strictly-positive tensor.** A post-ReLU activation quantized
  symmetric wastes every negative code; on this page's tensor that kind of mismatch costs about
  5% of the representable span even on data that is only mildly skewed, and it gets worse as
  skew increases.
- **Reporting `max_abs_err` alone.** One outlier element can dominate the max while every other
  element is quantized fine; `sym_mse_diff`/`asym_mse_diff`-style aggregate metrics catch a
  systematically-wrong scale that a single max value can hide.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists this
whole area as **116 tasks, adjacent-only elsewhere** — meaning every other resource found is
reference material or a course, and nothing outside this bank auto-grades int4/int8/int16
range and error the way the task above does:

- **[DeepLearning.AI — Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)**
  is the closest curriculum match: you build the same symmetric/asymmetric, per-tensor
  quantizer from scratch. It tops out at 13 code examples, and the one graded assignment is
  paid-tier only — this page's exercise is free and auto-checked.
- **[Maxime Labonne — Introduction to Weight Quantization](https://maximelabonne.substack.com/p/introduction-to-weight-quantization-2494701b9c0c)**
  implements the same absmax and zero-point INT8 math end to end on GPT-2 and is genuinely
  better at showing *why* it matters, comparing real perplexity before and after. It has no
  test harness — you read the printed output and judge it yourself, and it stops at plain
  INT8 rather than covering int4/int16 too.
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** is the production
  reference for the blockwise and NF4 schemes this page's int4 row only sketches — real dequant
  kernels behind a drop-in layer, not something you can practise against, but the ground truth
  once you want the next level of realism.
- **[ggml-org/llama.cpp's ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)**
  shows the actual packed-byte layout for GGUF's Q4_0/Q8_0 formats, which is the real version of
  the toy struct in the [bit-field-packing task](../tasks/cpp-bit-field-packed-int4-int8-quant-record/task.md)
  above.

## References

1. Google, *Quantization and Training of Neural Networks for Efficient
   Integer-Arithmetic-Only Inference* (Jacob et al., 2017) — the affine (asymmetric)
   quantization scheme this page's error table implements.
   https://arxiv.org/abs/1712.05877
2. PyTorch, *Quantization documentation* — `torch.qint8`, `torch.quint8` and the affine
   quantization parameters used in production.
   https://pytorch.org/docs/stable/quantization.html
