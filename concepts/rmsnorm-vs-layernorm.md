---
title: "What is layer normalization, and how does RMSNorm differ?"
description: "RMSNorm vs LayerNorm explained, with exact parameter/FLOP counts and a measured max-abs-difference table showing exactly what mean-subtraction buys, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is layer normalization, and how does RMSNorm differ?

RMSNorm vs LayerNorm is a choice of which row statistic a transformer block pays to
compute before its learned gain: LayerNorm subtracts the mean and divides by the standard
deviation, RMSNorm skips the mean and divides by the root-mean-square instead. At a
4,096-wide hidden state that missing mean step costs LayerNorm exactly 4,096 more
parameters and double the memory reads of RMSNorm per forward pass. Below, both counts
measured exactly, and the point — a zero row mean — where the two stop disagreeing at all.

## How LayerNorm and RMSNorm work

Both operations fight the same problem: activations drift in scale through a deep network,
and without renormalization a residual stream's variance explodes or collapses over many
layers. LayerNorm, from Ba, Kiros and Hinton (reference 1), treats each row as a small
sample: compute its mean, subtract it, divide by the standard deviation, then apply a
learned per-feature gain `gamma` and bias `beta`. RMSNorm, from Zhang and Sennrich
(reference 2), keeps only the rescaling half — it measures the root mean square of the raw
row, `sqrt(mean(x**2) + eps)`, and divides by that, with no mean subtraction and no bias, on
the claim that re-centering contributes little and variance-control does the real work.

That one missing step removes an entire reduction pass: LayerNorm sums the row once for the
mean, then sums squared deviations from that mean in a second pass, while RMSNorm's sum of
squares needs only one. On a GPU this is not an abstraction — both run as one block per row,
reduced by a warp-level tree exactly like
[the warp-shuffle butterfly reduction over the hidden dimension](../tasks/gpu-warp-reduction-layernorm-over-the-hidden-dim/task.md),
and reading that row efficiently is the same
[memory-coalescing](memory-coalescing.md) problem as any other kernel — a transposed or
strided layout turns a cheap reduction into an expensive one before the arithmetic starts. A
production kernel also has to decide whether to accumulate the sum of squares in the input's
own precision or upcast first; that is exactly the trade-off
[bfloat16 vs float16](bfloat16-vs-float16.md) exists to make legible, graded directly in
[fp32 upcast vs fp16 in-place RMSNorm](../tasks/llm-fp32-upcast-vs-fp16-in-place-rmsnorm/task.md).

Two consequences follow from RMSNorm's gain being a pure rescale. First, its per-feature
`gamma` folds directly into the next linear layer's weight matrix at inference —
[absorbing RMSNorm's gain into the next
matmul](../tasks/llm-weight-absorbed-rmsnorm-fold-gamma-into-next-matmul/task.md) removes
the elementwise multiply, which LayerNorm's additive bias cannot do. Second, neither
statistic says where in the block normalization sits — pre-norm, post-norm, or sandwich
wirings are an orthogonal axis, checked independently by
[classifying pre/post/sandwich norm wirings](../tasks/llm-classify-pre-post-sandwich-norm-wirings/task.md).

## LayerNorm vs batchnorm: which axis gets normalized

RMSNorm and LayerNorm both average over the hidden (feature) axis. Batch normalization
averages over a different axis, which is the real contrast behind layer normalization vs
batch normalization: BatchNorm computes, per feature, a mean and variance across the
mini-batch, then at inference swaps those live statistics for a running average tracked
during training. LayerNorm computes, per example, a mean and variance across that example's
own hidden features, with no dependence on any other row. That axis flip is why
transformers use LayerNorm or RMSNorm and rarely BatchNorm: autoregressive decoding
produces one token, hence a batch of one, at a time, and sequences vary in length, so there
is no stable batch statistic to average — a row's own feature statistic works identically
whether one sequence is running or a thousand.

In PyTorch, `torch.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)` is
this row-wise formula directly: it normalizes the last `len(normalized_shape)` dimensions of
every input independently and, when `elementwise_affine=True`, learns a `weight` (this
page's `gamma`) and `bias` (`beta`) — its default `eps` matches the `1e-5` used above.
`torch.nn.BatchNorm1d` instead carries `running_mean`/`running_var` buffers from training
and substitutes them in `.eval()` mode, a stateful step LayerNorm has no equivalent of,
since its statistics are recomputed fresh from every input regardless of mode.

## Parameters, operations, and where the outputs coincide

The first table counts, for one row of hidden size `d`, LayerNorm's and RMSNorm's learnable
parameters (`2d` versus `d`, gain-plus-bias versus gain-only) and their FLOPs and memory
reads by standard per-row-reduction accounting — `6d+1` FLOPs and `6d` reads for LayerNorm's
mean-then-variance passes, `4d+1` FLOPs and `3d` reads for RMSNorm's single pass — at `n=1`
row to isolate per-token cost. The second table feeds both formulas a seeded input and
reports the maximum absolute difference between outputs, on the input as generated and
after forcing each row's mean to exactly zero.

| hidden size d | LayerNorm params (2d) | RMSNorm params (d) | LayerNorm FLOPs (6d+1) | RMSNorm FLOPs (4d+1) | FLOPs ratio | LayerNorm mem reads (6d) | RMSNorm mem reads (3d) |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 64 | 385 | 257 | 1.4981 | 384 | 192 |
| 768 | 1,536 | 768 | 4,609 | 3,073 | 1.4998 | 4,608 | 2,304 |
| 4,096 | 8,192 | 4,096 | 24,577 | 16,385 | 1.5000 | 24,576 | 12,288 |
| 8,192 | 16,384 | 8,192 | 49,153 | 32,769 | 1.5000 | 49,152 | 24,576 |
| 65,536 | 131,072 | 65,536 | 393,217 | 262,145 | 1.5000 | 393,216 | 196,608 |

| input | max \|LayerNorm − RMSNorm\| |
|---|---|
| raw, seeded, nonzero row means | 1.488646 |
| same input, row mean forced to 0 | 6.661e-16 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
def norm_costs(d, n=1):
    params_ln, params_rms = 2 * d, d
    flops_ln, flops_rms = 6 * n * d + n, 4 * n * d + n
    mem_ln, mem_rms = 6 * n * d, 3 * n * d
    return params_ln, params_rms, flops_ln, flops_rms, mem_ln, mem_rms

print("-- parameters and operations, n=1 token --")
for d in (64, 768, 4096, 8192, 65536):
    p_ln, p_rms, f_ln, f_rms, m_ln, m_rms = norm_costs(d)
    print(f"d={d:>6} params_ln={p_ln:>7} params_rms={p_rms:>6} "
          f"flops_ln={f_ln:>7} flops_rms={f_rms:>7} flop_ratio={f_ln/f_rms:.4f} "
          f"mem_ln={m_ln:>7} mem_rms={m_rms:>6} mem_ratio={m_ln/m_rms:.1f}")

print("-- where the outputs actually differ --")
import numpy as np

def layernorm(x, gamma, beta, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rmsnorm(x, gamma, eps):
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return gamma * x / np.sqrt(ms + eps)

rng = np.random.default_rng(0)
B, D = 4, 8
x = rng.normal(loc=5.0, scale=2.0, size=(B, D))       # deliberately nonzero row means
gamma = rng.normal(size=D)
beta = np.zeros(D)                                     # isolate mean-subtraction, not beta
eps = 1e-5

diff_raw = float(np.max(np.abs(layernorm(x, gamma, beta, eps) - rmsnorm(x, gamma, eps))))
x0 = x - x.mean(axis=-1, keepdims=True)                # force each row's mean to exactly 0
diff_centered = float(np.max(np.abs(layernorm(x0, gamma, beta, eps) - rmsnorm(x0, gamma, eps))))
print(f"diff_raw={diff_raw:.6f}")
print(f"diff_centered={diff_centered:.3e}")
PY
```

Read the first table as two different curves hiding in one folklore number. The memory-read
ratio is exactly **2.0** at every width, because neither `6d` nor `3d` carries the extra
`+1` that FLOPs do — RMSNorm's halved memory traffic is exact from `d=64` up, not an
asymptotic property. The FLOPs ratio, by contrast, only reaches the textbook **1.5x** in
the limit: at `d=64` it is **1.4981**, and it does not round to 1.5000 until `d=4,096`. The
second table is the sharper result: on a row with an ordinary nonzero mean, the two
formulas disagree by **1.488646** at their largest element — a real difference, not
rounding. Force that row's mean to exactly zero and the gap collapses to **6.661e-16**,
floating-point noise, because when `mean(x) = 0`, `mean(x**2)` *is* the variance, so RMS and
standard deviation coincide and the two formulas become algebraically identical
(`beta=0`). Mean-subtraction buys nothing once the input is already centered — it only
matters when it isn't.

## Practise it

```bash
mlsys grade llm-debug-accidental-mean-centering-in-rmsnorm
```

[That task](../tasks/llm-debug-accidental-mean-centering-in-rmsnorm/task.md) gates
`max_abs_err <= 1e-06` against a reference RMSNorm. The shipped starter centers the row
before dividing by its RMS — exactly the bug the second table isolates — so on any input
whose row mean isn't already near zero, it fails by roughly the same **1.488646** margin
measured above, not a rounding error.

In roughly increasing difficulty:
[RMSNorm from scratch, Llama style](../tasks/llm-rmsnorm-from-scratch-llama-style/task.md)
(gates a numeric Jacobian too),
[FLOP and memory savings, RMSNorm vs LayerNorm](../tasks/llm-flop-memory-savings-rmsnorm-vs-layernorm/task.md)
(the formulas the first table uses),
[LayerNorm forward from scratch](../tasks/llm-layernorm-forward-from-scratch/task.md),
[RMSNorm forward inside a CUDA kernel](../tasks/gpu-rmsnorm-forward/task.md), and
[RMSNorm backward from scratch](../tasks/llm-rmsnorm-backward-from-scratch/task.md), where
the missing mean term changes the gradient too.

## Common mistakes

- **Assuming the two are numerically interchangeable on real activations.** They agree to
  the last bit only once the row is zero-mean; on the seeded, ordinary-mean input above they
  disagree by 1.488646 at their largest element, not by rounding noise.
- **Copying a LayerNorm implementation and forgetting to delete the centering line.** The
  primary exercise grades exactly this: `x - mean(x)` silently survives the port and still
  returns plausible numbers — wrong by the same margin as the raw-input row above.
- **Quoting "1.5x FLOPs, 2x memory" as if both were the same kind of number.** The memory
  ratio is exactly 2.0 at every hidden size measured; the FLOPs ratio is an asymptote the
  formula only reaches at d=4,096 and above — at d=64 it is 1.4981, not 1.5.
- **Folding RMSNorm's trick onto LayerNorm.** The gain absorbs into the next matmul because
  RMSNorm's rescale is diagonal; LayerNorm's additive bias and mean-subtraction don't
  collapse into a weight matrix the same way.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md)'s LLM-internals
section, which calls this whole area **"Crowded"**:

- **[Stanford CS336, Assignment 1](https://github.com/stanford-cs336/assignment1-basics)** —
  `tests/adapters.py` requires `run_rmsnorm` checked against a reference under pytest. Free
  and real; it does not isolate mean-subtraction or count parameters and FLOPs.
- **[Build a Large Language Model (rasbt/LLMs-from-scratch)](https://github.com/rasbt/LLMs-from-scratch)**
  — builds LayerNorm chapter-by-chapter, GPT-2 style rather than RMSNorm.
- **[ARENA 3.0, Chapter 1](https://github.com/callummcdougall/ARENA_3.0)** — builds a
  GPT-2-style transformer including layer norm from scratch, checked against intermediate
  tensors; the most actively maintained resource here, and it skips RMSNorm.
- **[Deep-ML — Attention Is All You Need collection](https://www.deep-ml.com/collections/Attention%20Is%20All%20You%20Need)**
  — a browser-graded "layer norm for sequences" problem among six; grades the forward pass
  alone.
- The survey's verdict: every resource above checks final-tensor correctness against a
  reference; none grade a structural property, which is the gap the debug task fills.

## References

1. Ba, J. L., Kiros, J. R., Hinton, G. E., *Layer Normalization*, 2016.
   https://arxiv.org/abs/1607.06450
2. Zhang, B., Sennrich, R., *Root Mean Square Layer Normalization*, NeurIPS 2019.
   https://arxiv.org/abs/1910.07467
