---
title: "GGUF vs safetensors: what is the difference?"
description: "GGUF vs safetensors explained, with a measured bytes-per-weight table (real bit-packing, not the headline bit count) you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# GGUF vs safetensors: what is the difference?

GGUF vs safetensors is a choice between two on-disk container formats for model weights:
safetensors stores each tensor at its native dtype behind a small JSON header, while GGUF
bundles the tensors with a block quantization scheme that packs some of them below one byte per
weight. That packing is not free — a GGUF `Q4_0` block advertises 4 bits per weight but actually
costs 4.5 once its scale factor is counted, and the "smarter" `Q4_K` format costs exactly the
same 4.5, for a real reason measured below.

## How it works

A safetensors file is a JSON header — tensor names, dtypes, shapes, byte offsets — followed by
one flat buffer of raw values. There is no pickle, no arbitrary code path, and no compute graph;
loading it is a `mmap` and a dtype cast, which is also why it replaced PyTorch's `.bin` checkpoints
as the default: a `.bin` file could execute code on load, a safetensors file cannot. Whatever dtype
the tensor was saved in — fp32, fp16, [bfloat16](bfloat16-vs-float16.md), even a pre-packed int8 —
safetensors stores it verbatim. It has no opinion about compression.

GGUF, from `llama.cpp`, does have an opinion. It is a single-file container for architecture
metadata plus tensors, where each tensor's declared type can itself be a block quantization
scheme rather than a dtype: `F16`, `Q8_0`, `Q4_0`, `Q4_K`, and others. The legacy schemes split a
tensor into fixed-size blocks of 32 values, compute one shared scale per block — usually
`max(|w|) / 7` for the signed 4-bit case — and pack every value's code at less than a byte: two
4-bit codes per byte for `Q4_0`, one signed byte for `Q8_0`. This is the same symmetric-range
machinery as [int8/int4 quantization ranges](integer-quantization-ranges.md), just applied per
32-element block instead of per tensor, which is what keeps the error local instead of being set
by the single largest weight anywhere in the row.

The "k-quant" formats add a second level. `Q4_K` groups eight 32-value sub-blocks into one
256-value super-block, gives each sub-block its own *scale and minimum* (an asymmetric range, not
just a symmetric one), and then compresses those sixteen numbers into 6-bit codes relative to two
super-block-level fp16 constants — a quantized codebook describing the codebook, [python's
`__slots__`](python-slots.md)-style fixed-overhead-per-unit reasoning one level up. That
extra machinery is exactly the part a "4-bit" label leaves silent, and it is why `Q4_0` and
`Q4_K` need to be measured in real packed bytes rather than trusted at face value — the same
"count what the format actually gives you, not what the name promises" instinct behind measuring
[memory coalescing](memory-coalescing.md) in transactions instead of trusting "one instruction, one
read." GGUF's `mmap`-and-run design is also why the weights it stores are usually already
irreversibly quantized: there is no bf16 fallback tensor sitting next to the Q4 one, so whatever
error this page measures is the error a `Q4_0` or `Q4_K` GGUF file ships with, permanently.

## Bytes per weight measured, not assumed

Nominal bit width was fixed by each format's name. What was actually measured, for a fixed
`N(0,1)` tensor of 4,096 float32 values (`numpy.random.default_rng(0)`), is the real packed byte
count per format — codes plus every scale and minimum, bit-for-bit — and the resulting
reconstruction error against the original tensor.

| format | bits/weight nominal | bytes/weight actual | max abs err |
|---|---|---|---|
| fp16 (safetensors) | 16 | 2.0000 | 0.000974 |
| Q8_0 | 8 | 1.0625 | 0.015233 |
| Q4_0 | **4** | **0.5625** | 0.277829 |
| Q4_K-like | **4** | **0.5625** | 0.188414 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

rng = np.random.default_rng(0)
N = 4096
x = rng.normal(size=N).astype(np.float32)

def fp16_roundtrip(x):
    q = x.astype(np.float16)
    return q.nbytes, q.astype(np.float32)

def q8_0_roundtrip(x, block=32):
    xb = x.reshape(-1, block)
    amax = np.abs(xb).max(axis=1)
    d = np.where(amax == 0, 0.0, amax / 127.0).astype(np.float32)
    ds = np.where(d == 0, 1.0, d)
    q = np.clip(np.round(xb / ds[:, None]), -127, 127).astype(np.int8)
    xhat = (q.astype(np.float32) * d[:, None]).reshape(-1)
    return q.nbytes + d.astype(np.float16).nbytes, xhat

def q4_0_roundtrip(x, block=32):
    xb = x.reshape(-1, block)
    amax = np.abs(xb).max(axis=1)
    d = np.where(amax == 0, 0.0, amax / 7.0).astype(np.float32)
    ds = np.where(d == 0, 1.0, d)
    q = np.clip(np.round(xb / ds[:, None]), -8, 7).astype(np.int8)
    xhat = (q.astype(np.float32) * d[:, None]).reshape(-1)
    packed = xb.shape[0] * (block // 2)          # 2 nibble codes per byte
    return packed + d.astype(np.float16).nbytes, xhat

def pack_6bit(vals):
    out = bytearray()
    for i in range(0, vals.size, 4):
        v0, v1, v2, v3 = (int(v) for v in vals[i:i + 4])
        out += (v0 | v1 << 6 | v2 << 12 | v3 << 18).to_bytes(3, "little")
    return bytes(out)

def q4k_like_roundtrip(x, sub=32, superb=256):
    ns, spb = x.size // superb, superb // sub
    xb = x.reshape(ns, spb, sub)
    smin, smax = xb.min(2), xb.max(2)
    sscale = (smax - smin) / 15.0
    d = sscale.max(1) / 63.0
    dmin = np.abs(np.minimum(smin, 0.0)).max(1) / 63.0
    ds, dms = np.where(d == 0, 1.0, d), np.where(dmin == 0, 1.0, dmin)
    s_code = np.clip(np.round(sscale / ds[:, None]), 0, 63).astype(np.uint8)
    m_code = np.clip(np.round(np.maximum(-smin, 0.0) / dms[:, None]), 0, 63).astype(np.uint8)
    scale_hat, min_hat = s_code * d[:, None], m_code * dmin[:, None]
    shs = np.where(scale_hat == 0, 1.0, scale_hat)
    q = np.clip(np.round((xb + min_hat[:, :, None]) / shs[:, :, None]), 0, 15).astype(np.uint8)
    xhat = (q * scale_hat[:, :, None] - min_hat[:, :, None]).reshape(-1)
    code_bytes = sum(((qf[0::2] | (qf[1::2] << 4)).astype(np.uint8)).nbytes
                      for qf in q.reshape(ns, superb))
    sm_bytes = sum(len(pack_6bit(np.concatenate([s_code[i], m_code[i]]))) for i in range(ns))
    return code_bytes + sm_bytes + d.astype(np.float16).nbytes + dmin.astype(np.float16).nbytes, xhat

for name, fn, nominal in [("fp16", fp16_roundtrip, 16), ("Q8_0", q8_0_roundtrip, 8),
                           ("Q4_0", q4_0_roundtrip, 4), ("Q4_K-like", q4k_like_roundtrip, 4)]:
    nbytes, xhat = fn(x)
    print(f"{name} nominal_bits={nominal} bytes_per_weight={nbytes / N:.4f} "
          f"max_abs_err={np.abs(x - xhat).max():.6f}")
PY
```

Two things the table says that the "4 bits" headline does not. First, the overhead is a flat
extra 0.5 bits per weight for a 32-element block regardless of code width — one shared fp16 scale
divided by 32 values is `16/32 = 0.5` bits either way — so it is proportionally worse the
narrower the code: `+12.5%` on Q4's 4 bits, only `+6.25%` on Q8's 8. Second, `Q4_0` and
`Q4_K-like` land on the *identical* 0.5625 bytes/weight, because k-quant's extra per-sub-block
minimum is paid for by amortizing its own 6-bit-compressed metadata over a 256-value super-block
instead of a 32-value one — and for that identical budget it buys a 32% lower max error (0.188 vs
0.278) by giving each 32-value group its own offset instead of one symmetric scale for the whole
row. `fp16` has no block at all, so its actual cost equals its nominal cost exactly: safetensors'
"waste" is 2.0 bytes it never hides, GGUF's efficiency is 0.5625 bytes it never states.

## Practise it

```bash
mlsys grade cpp-gguf-style-q4-0-block-pack-unpack-round-trip
```

[That task](../tasks/cpp-gguf-style-q4-0-block-pack-unpack-round-trip/task.md) gates a real
`pack_q4_0`/`unpack_q4_0` C++ implementation on `exact_match == 1.0` against a fixed reference: the
18-byte struct above, packed and unpacked byte-for-byte. The shipped starter writes nothing to the
output buffer and returns all zeros from unpack, so it fails immediately; the harder failure mode
is a correct scale and clip with the nibble order swapped — `qs[i] = (q[i+16]+8) | ((q[i]+8)<<4)`
instead of the reverse — which still round-trips to the right *floats* on a single block but packs
different bytes, and `exact_match` catches that where a numeric tolerance would not.

More of the same format, in increasing difficulty:
[compute effective bits per weight including scale overhead](../tasks/rwc-effective-bits-weight-including-scale-bias-overhead/task.md)
(the formula behind the 4→4.5 gap above, for arbitrary block size),
[Q8_0 block quantization](../tasks/rwq-q8-0-block-quantization-32-elem-int8/task.md),
[Q8_0 dequant matching ggml exactly](../tasks/rwc-q8-0-block-quant-dequant-matches-ggml/task.md),
[match a bits-per-weight budget to a GGUF type](../tasks/rwq-match-a-bits-per-weight-budget-to-a-gguf-type/task.md),
and the real thing this page's `Q4_K-like` function only approximates:
[Q4_K two-level super-block quantization](../tasks/rwc-q4-k-super-block-256-two-level-k-quant/task.md).

## Common mistakes

- **Reading "4-bit" as 4 bits per weight on disk.** The table shows `Q4_0` actually costs 0.5625
  bytes/weight — 4.5 bits, 12.5% over the name — and that gap is structural, not a rounding
  artifact: it is the block's one shared scale, divided by the block size.
- **Assuming a fancier scheme costs more.** `Q4_K` stores a scale *and* a minimum per 32-value
  sub-block, twice the metadata of `Q4_0`'s single scale, and still lands on the same 0.5625
  bytes/weight, because that metadata is itself compressed and shared across a 256-value
  super-block instead of paid per 32-value block.
- **Treating "GGUF" and "quantized" as synonyms.** Safetensors is dtype-agnostic — GPTQ and AWQ
  checkpoints ship packed int4 weights inside safetensors files routinely. The actual
  differentiator is that GGUF's block scheme is fixed by the container format itself; safetensors
  has no opinion on what bytes it holds.
- **Comparing formats by nominal bit width and expecting equal accuracy.** `Q4_0` and `Q4_K-like`
  spend the identical 0.5625 bytes/weight here and land 32% apart on max error, because *how* a
  budget is spent (one symmetric scale vs. per-32 asymmetric scale-and-min) matters as much as
  the budget itself.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists this whole
area as **116 tasks, adjacent only** — meaning everything else found is reference code or a course,
not something with a starter/reference split to fail against:

- **[ggml-org/llama.cpp — ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)**
  is the real bit-packing source this page's `Q4_0`/`Q4_K-like` functions approximate — the actual
  super-block scale quantization and byte layout, C production code with no exercises attached.
- **[Maxime Labonne's LLM Course — quantization notebooks](https://github.com/mlabonne/llm-course)**
  is the roadmap most people actually follow to go from a safetensors checkpoint to a GGUF file:
  runnable Colabs applying `llama.cpp`'s own converter and inspecting the resulting file size, not
  implementing the packing yourself.
- **[DeepLearning.AI — Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)**
  has you build a symmetric/asymmetric, per-tensor/per-group quantizer from scratch in PyTorch —
  the mechanism this page's block math is built from — but its one graded assignment is paid-tier
  only, and it stops before GGUF's two-level super-blocks.
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** is the production
  reference for NF4 and blockwise 8-bit quantization, the safetensors-side cousin of GGUF's block
  scheme — real dequant kernels behind a drop-in layer, nothing to submit against.

## References

1. ggml-org, *GGUF file format specification*. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
2. ggml-org/llama.cpp, `ggml-quants.c` — reference `block_q4_0`/`block_q8_0`/`block_q4_K` layouts
   this page's byte counts match exactly (18, 34, and 144 bytes respectively).
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
3. Hugging Face, *Safetensors* — format documentation and the pickle-execution rationale for its
   design. https://huggingface.co/docs/safetensors/index
