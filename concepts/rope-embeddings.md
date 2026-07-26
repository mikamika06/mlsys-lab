---
title: "What are rope embeddings?"
description: "RoPE embeddings explained, with position-difference invariance verified in float64, measured against naive absolute encodings and long-context extrapolation, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What are rope embeddings?

RoPE embeddings rotate a query or key vector's paired dimensions by an angle set only by
that vector's absolute position, so the dot product of a rotated query and key ends up
depending only on how far apart the two positions are — never on either position alone. That
difference-only property holds to within 1e-11 of float64 rounding even a million positions out
positions, while the same two vectors under an additive absolute position encoding swing by
3.056078 across that identical fixed gap. Both measured below, with what happens to RoPE's own
guarantee once a sequence runs past the length the model was trained on.

## How it works

RoPE splits a head's d-dimensional vector into d/2 two-dimensional pairs and rotates pair *i*
by an angle θᵢ·pos, where θᵢ = base^(−2i/d) and base is conventionally 10,000. Low-index pairs
turn fast — θ₀ is 1 radian per position step — and high-index pairs turn slowly, so one
rotation carries both fine local order and coarse long-range order, at no cost in learned
parameters: there is no embedding table to look up, unlike an absolute scheme.

### Rotary position embedding, in one identity

The relative-position guarantee follows from one fact about 2D rotations: rotating by θm then
dotting against a rotation by θn is the same as rotating the difference θm − θn, since rotation
matrices compose by angle subtraction (R(θn)ᵀR(θm) = R(θm − θn)). Apply that per pair and sum,
and the whole vector's post-rotation dot product becomes a function of m − n alone — one
trigonometric identity, applied independently to d/2 pairs, and why rotary position embedding
displaced learned absolute position tables in essentially every open LLM built after 2022.

The rotation applies to queries and keys only, never values, and runs in fp32 even inside a
[bfloat16](bfloat16-vs-float16.md) model: θᵢ·pos grows large as `pos` grows, and a narrower
sine/cosine loses the precision position needs — the same "which format actually holds the
value" question [gguf vs safetensors](gguf-vs-safetensors.md) asks of weight storage, and the
same float64 floor [Kahan summation](kahan-summation.md) measures directly.

Two incompatible conventions exist for which dimensions pair up: interleaved, (2i, 2i+1), as in
the original paper, versus the "half-split" layout most open implementations ship instead,
where dimension *i* pairs with *i + d/2*. One convention's code on the other's trained weights
scrambles every learned position without touching the loss — measured below as an L2 distance
of 0.647127 between the two conventions' output on the same vector. Neither
[softmax vs sigmoid](softmax-vs-sigmoid.md) nor [continuous batching](continuous-batching.md)
touches position — RoPE is its own axis of the block. Where it meets system state is the KV
cache: re-deriving position ids wrong on append is a bug class two tasks below gate directly,
the same failure [gradient checkpointing](gradient-checkpointing.md)'s recomputation schedule
guards against for an unrelated reason.

## Position-difference invariance measured, and what breaks it

A fixed 8-dimensional float64 query and key (`numpy.random.default_rng(0)`) are dotted at
positions (5, 2) — a difference of 3 — then both positions are shifted together by 0 up to
1,000,000. Counted: the RoPE dot product's drift from its own shift-0 value, and the same two
vectors' dot product under an additive sinusoidal absolute position encoding instead.

| shift added to both positions | RoPE dot product | drift from shift = 0 | additive absolute-PE dot product |
|---|---|---|---|
| 0 | -1.586475 | 0 | 1.176863 |
| 1 | -1.586475 | True | -0.202496 |
| 10 | -1.586475 | 0 | 0.780650 |
| 100 | -1.586475 | True | -0.574756 |
| 10,000 | -1.586475 | True | 1.710141 |
| 1,000,000 | -1.586475 | True | 2.481322 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def rope_rotate(x, pos, base=10000.0):
    """Interleaved convention: pairs are (0,1), (2,3), ..."""
    d = x.shape[-1]
    idx = np.arange(0, d, 2)
    inv_freq = 1.0 / (base ** (idx / d))
    theta = pos * inv_freq
    cos, sin = np.cos(theta), np.sin(theta)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out

def rope_half_split(x, pos, base=10000.0):
    """Half-split convention: pairs are (i, i + d/2)."""
    d = x.shape[-1]
    half = d // 2
    idx = np.arange(half)
    inv_freq = 1.0 / (base ** (2 * idx / d))
    theta = pos * inv_freq
    cos, sin = np.cos(theta), np.sin(theta)
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out

def sinusoidal_abs_pe(pos, d, base=10000.0):
    i = np.arange(d)
    angle = pos / (base ** (2 * (i // 2) / d))
    pe = np.empty(d)
    pe[0::2] = np.sin(angle[0::2])
    pe[1::2] = np.cos(angle[1::2])
    return pe

rng = np.random.default_rng(0)
d = 8
q = rng.normal(size=d).astype(np.float64)
k = rng.normal(size=d).astype(np.float64)
m0, n0 = 5, 2  # fixed difference Delta = 3

base_dot = float(np.dot(rope_rotate(q, m0), rope_rotate(k, n0)))
abs_dots = []
for shift in (0, 1, 10, 100, 10_000, 1_000_000):
    m, n = m0 + shift, n0 + shift
    rope_dot = float(np.dot(rope_rotate(q, m), rope_rotate(k, n)))
    abs_dot = float(np.dot(q + sinusoidal_abs_pe(m, d), k + sinusoidal_abs_pe(n, d)))
    abs_dots.append(abs_dot)
    print(f"shift={shift:<9} rope_dot={rope_dot:.6f}  "
          f"rope_diff_under_1e-11={abs(rope_dot - base_dot) < 1e-11!s:<5} "
          f"abs_pe_dot={abs_dot:.6f}")
print(f"abs_pe_swing={max(abs_dots) - min(abs_dots):.6f}")

# interleaved vs half-split: same q, same base, same position -- different vectors
conv_mismatch = float(np.linalg.norm(rope_rotate(q, 7) - rope_half_split(q, 7)))
print(f"convention_mismatch_l2={conv_mismatch:.6f}")

# extrapolation past the trained context, and the position-interpolation fix
d2, base2 = 128, 10000.0
L_train, L_target = 4096, 8192
idx2 = np.arange(0, d2, 2)
inv_freq2 = 1.0 / (base2 ** (idx2 / d2))
period = 2 * np.pi / inv_freq2                       # positions per full rotation, per channel
n_channels = len(inv_freq2)
never_full_rotation_train = int(np.sum(period > L_train))
still_unseen_at_target = int(np.sum(period > L_target))

slow_theta = inv_freq2[-1]                            # the slowest-rotating channel
train_max_phase = slow_theta * (L_train - 1)
target_max_phase_noscale = slow_theta * (L_target - 1)
s = L_train / L_target                                # linear position-interpolation scale
target_max_phase_pi = slow_theta * (L_target - 1) * s

print(f"channels={n_channels}  never_full_rotation_at_L_train={never_full_rotation_train}  "
      f"still_unseen_at_L_target={still_unseen_at_target}")
print(f"train_max_phase={train_max_phase:.4f}  "
      f"noscale_phase={target_max_phase_noscale:.4f} ({target_max_phase_noscale/train_max_phase:.4f}x)  "
      f"pi_phase={target_max_phase_pi:.4f} ({target_max_phase_pi/train_max_phase:.4f}x)")
PY
```

RoPE's dot product is identical to six decimals at every shift and drifts by no more than
under 1e-11 even after a million-position shift — float64 rounding, not a signal leak. The residual is last-bit noise and differs between numpy kernels on x86 and ARM, so only the bound is quoted. The
additive encoding has no such floor: at the identical fixed difference of 3 it ranges from
-0.574756 to 2.481322, a swing of 3.056078 on vectors whose own dot product is order 1. Nothing
about "add a position vector" guarantees relative structure; RoPE's guarantee comes specifically
from rotating, because rotation composition is where the absolute position cancels.

## Extrapolation past the trained context, and the position-interpolation fix

Same run, continued: a realistic head dimension of 128 (base 10,000) gives 64 rotary frequency
channels. For a model trained at L_train = 4,096, count channels that never complete one full
2π rotation inside that range, then compare the phase the slowest channel reaches at a naively
doubled context (L_target = 8,192) against the phase position interpolation reaches once
positions are scaled by L_train / L_target first.

| quantity | value |
|---|---|
| frequency channels (d/2, at d=128) | 64 |
| channels that never complete a full rotation within L_train=4,096 | 18 |
| channels still short of a full rotation even at L_target=8,192 | 14 |
| slowest channel's max phase seen during training | 0.4729 rad |
| same channel's phase at L_target−1, no scaling | 0.9459 rad (2.0002× the trained max) |
| same channel's phase at L_target−1, with position interpolation | 0.4729 rad (1.0001× the trained max) |

18 of 64 channels — over a quarter — never trace a full circle during training, so doubling the
context naively pushes the slowest one to a phase 2.0002 times larger than anything rehearsed.
Position interpolation's fix is the same scale-by-L_train/L_target step in the snippet above:
it pulls that phase back to within 1.0001× of the trained maximum, at the cost of halving the
angular gap between adjacent tokens everywhere, including channels that were already fine.
Trading resolution for range is why frequency-aware schemes like NTK-scaling and YaRN exist
instead; neither is modelled here.

## Practise it

```bash
mlsys grade llm-apply-rope-to-q-k-match
```

[That task](../tasks/llm-apply-rope-to-q-k-match/task.md) gates a vectorised
`apply_rope(x, pos)` on `max_abs_err <= 1e-06` against a NumPy reference; `check.py` folds a
shape or dtype mismatch into that same metric as `inf`. The starter raises `NotImplementedError`
immediately, so it fails trivially — the sharper trap is that this task's per-pair frequency is
`ωᵢ = linspace(0.01, 0.99, d/2)`, linearly spaced, not the geometric `base^(−2i/d)` schedule
used above. Porting this page's formula verbatim passes the rotation algebra and still fails the
gate: "RoPE" names a family of frequency schedules, not one fixed function.

More of the same mechanism, in increasing scope:
[relative-position invariance as a function](../tasks/llm-rope-relative-position-invariance/task.md)
(`rel_err <= 1e-4`),
[vectorised RoPE under a line-count budget](../tasks/llm-vectorized-rope-no-python-loop/task.md)
(`max_abs_err <= 1e-6`, `line_count <= 40`),
[the complex-number equivalence](../tasks/llm-complex-number-rope-equivalence/task.md)
(`max_abs_err <= 1e-7`),
[debugging interleaved vs half-split pairing](../tasks/llm-debug-interleaved-vs-half-split-rope-convention/task.md)
(`max_abs_err < 1e-6`, the mismatch above as a bug to find),
[fusing the rotation into the QK score](../tasks/llm-fuse-rope-into-the-qk-score/task.md)
(`max_abs_err <= 1e-5`),
[linear position interpolation](../tasks/llm-linear-position-interpolation-pi/task.md)
(`max_abs_err <= 1e-6`),
[RoPE against ALiBi under extrapolation](../tasks/llm-alibi-vs-rope-extrapolation/task.md)
(`rel_err <= 0.01`),
[a fused QKV-plus-RoPE-plus-cache-write pass](../tasks/llm-fused-qkv-rope-kv-cache-write-in-one-pass/task.md)
(`max_abs_err <= 1e-5`),
and [RoPE applied twice on a reused cache entry](../tasks/sys-fix-double-applied-rope-on-cache-reuse/task.md)
(`rel_err <= 1e-5`).

## Common mistakes

- **Trusting an additive position encoding to preserve relative structure.** The table above
  measures a 3.056078 swing at a fixed position difference of 3: a scheme with no algebraic
  reason to be shift-invariant is not, however sinusoidal it looks.
- **Mixing interleaved and half-split pairing conventions.** The same vector, position, and
  base give outputs 0.647127 apart in L2 norm. Neither convention is wrong alone; one
  convention's code on the other's weights is.
- **Assuming RoPE's exact invariance protects long-context extrapolation.** The identity holds
  at any difference — it never degrades. What degrades is that 18 of 64 channels at a real
  128-dim head never complete one rotation during a 4,096-token training run, so the phase
  values a naively extended model reaches past that length were never rehearsed.
- **Treating "RoPE" as one fixed frequency formula.** The practice task above uses a
  linearly-spaced `ωᵢ`, not the geometric `10000^(−2i/d)` schedule measured here; a correct
  rotation with the wrong schedule still fails — a different, equally valid member of the same
  family.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md): RoPE *scaling* —
NTK-aware, YaRN, or the linear interpolation measured above, distinct from applying RoPE itself
— has no graded or self-checked exercise anywhere else found.

- **[LeetGPU — challenge set](https://leetgpu.com/challenges)** has a dedicated
  `rotary-positional-embedding` browser challenge among ~90 kernels, hidden-test graded;
  freemium, and skips scaling past the trained context.
- **[Stanford CS336 — Assignment 1](https://github.com/stanford-cs336/assignment1-basics)**
  requires a working `run_rope` among ten pytest-graded components of a from-scratch
  language model; free, but one line item in a larger assignment.
- **[Deep-ML — Attention Is All You Need collection](https://www.deep-ml.com/collections/Attention%20Is%20All%20You%20Need)**
  has a positional-encoding problem graded in-browser, freemium — the sinusoidal scheme this
  page measures RoPE against, not RoPE itself.
- **[The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)** covers
  that sinusoidal encoding — a runnable notebook, free, nothing to submit and
  nothing that checks your work.

## References

1. Su, J. et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021/2023.
   https://arxiv.org/abs/2104.09864
2. Chen, S. et al., *Extending Context Window of Large Language Models via Positional
   Interpolation*, 2023. https://arxiv.org/abs/2306.15595
3. EleutherAI, *Rotary Embeddings: A Relative Revisit* — the interleaved-vs-half-split
   convention split documented from the implementation side.
   https://blog.eleuther.ai/rotary-embeddings/
