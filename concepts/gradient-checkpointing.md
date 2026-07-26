---
title: "What is gradient checkpointing?"
description: "Gradient checkpointing explained, with a measured retained-activations-vs-segment-count table you can reproduce with plain arithmetic, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is gradient checkpointing?

Gradient checkpointing is a training-memory technique that discards most intermediate
activations right after the forward pass produces them and regenerates the missing ones from
a handful of saved checkpoints when the backward pass needs them. Splitting a 144-layer
network into 12 segments instead of one keeps peak retained activations at 24 instead of 145,
at the cost of exactly one extra full forward pass through all 144 layers. Below, that
trade-off is counted segment size by segment size, with no framework and no timer involved.

## How it works

Ordinary backpropagation keeps every layer's activation alive from the moment the forward pass
produces it until the backward pass consumes it, because the local gradient at each layer is a
function of that layer's input. For an `L`-layer network this means peak memory grows linearly
with depth — the entire forward graph is resident at the deepest point of backward.

Gradient checkpointing breaks that chain. Split the `L` layers into `S` contiguous segments and
keep only the activation entering each segment — `S` checkpoints total — discarding everything
produced inside a segment once it's no longer needed by the next layer. When backward reaches a
segment, it replays that segment's forward pass from its saved checkpoint, reconstructs the
segment's interior activations, uses them for the local backward step, then discards them again.
Every layer gets recomputed exactly once this way, no matter how the segments are drawn — so the
total extra forward work is always `L` layer-evaluations, one additional pass, not a multiple
that grows with `S`.

The only real knob is where to cut. Few, large segments keep few checkpoints (`S` is small) but
each backward step has to rebuild a large chunk of the network, so the peak working set —
checkpoints held plus the segment currently being replayed — is dominated by the segment size.
Many, small segments shrink that per-segment cost but need more checkpoints resident at once.
Both extremes are expensive for the same reason, and retained memory is minimized where the two
terms balance: `S` segments of size `L / S` each, with `S ≈ √L`.

This is the same trade as [cache blocking](cache-blocking.md): both pay extra work to keep a
working set small enough to be cheap, instead of materializing everything a fixed budget can't
hold. It differs from the layout family — [memory coalescing](memory-coalescing.md) and
[false sharing](false-sharing.md) — because those rearrange a *fixed* amount of data, while
checkpointing changes how much data exists at any one instant. It sits next to the per-element
story too: [bfloat16 vs float16](bfloat16-vs-float16.md) and
[int8/int4 quantization ranges](integer-quantization-ranges.md) shrink each stored value;
checkpointing shrinks the count retained, and real stacks combine both.
[Python `__slots__`](python-slots.md) is the same idea one level down, at a single object rather
than a computation graph.

The two failure modes that matter here are correctness bugs, not slowdowns: a replayed segment
must reuse the exact random state (dropout masks especially) the first forward pass used, or the
recomputed activations silently diverge from the real ones; and a segment boundary drawn through
a stateful op — a running-mean update, an in-place write — applies that mutation twice. Neither
raises an exception; the first symptom is a gradient that just doesn't match an uncheckpointed
run.

## Retained activations measured against segment count

Fixing a 144-layer network (a perfect square, so every `S` below divides it evenly), the only
thing varied is the segment count `S`. For each `S` the script counts checkpoints kept (`S`),
activations live in the segment being replayed (`s = L / S`), and total extra forward-layer
evaluations during backward.

| segments S | segment size s | retained activations | extra forward layers |
|---|---|---|---|
| 1 | 144 | 145 | 144 |
| 4 | 36 | 40 | 144 |
| 9 | 16 | 25 | 144 |
| 12 | 12 | **24** | 144 |
| 16 | 9 | 25 | 144 |
| 36 | 4 | 40 | 144 |
| 144 | 1 | 145 | 144 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
L = 144  # perfect square, so every S below divides it with no remainder
segment_counts = [1, 4, 9, 12, 16, 36, 144]

for S in segment_counts:
    s = L // S                       # segment size
    retained = S + s                 # S checkpoints + s activations in the live segment
    extra_forward = S * s            # every layer recomputed exactly once, however it's split
    print(f"S={S:>3}  s={s:>3}  retained={retained:>3}  extra_forward={extra_forward}")
PY
```

Read `retained` as a parabola in `S` that bottoms out exactly at `S = 12 = √144`, giving `2·√144
= 24`. Move off that point in either direction and the count rises symmetrically — `S = 9` and
`S = 16` both cost 25, `S = 4` and `S = 36` both cost 40, and the two extremes converge on `145`,
one *more* than storing every layer outright, because at `S = 1` or `S = 144` the model is
paying for both the checkpoint and the redundant working-set copy of the same data. Meanwhile
`extra_forward` never moves off `144`: whichever `S` is chosen, exactly `L` layer-evaluations get
redone somewhere, because the recomputation is always exactly one full forward pass, just cut
into different-sized pieces. The practical stopping point isn't the literal minimum — going from
`S = 9` to `S = 12` buys one retained activation, a 4% gain, so most real schedulers pick
whichever divisor of `L` lands nearest `√L` rather than searching for the exact optimum.

## Practise it

```bash
mlsys grade rwm-sqrt-l-segment-size-and-its-memory-recompute-cost
```

[That task](../tasks/rwm-sqrt-l-segment-size-and-its-memory-recompute-cost/task.md) gates
`exact_match == 1.0` against an oracle that returns
`(round(√L), 2·round(√L), L)` for eleven layer counts. The shipped starter raises
`NotImplementedError`, so it fails before any comparison happens; the more interesting way to
fail is truncating instead of rounding — `int(√L)` instead of `round(√L)` gives the same
`segment_size` as the oracle only when `L` sits exactly on a perfect square, and diverges for
everything else. (At `L = 99`, `√99 ≈ 9.95`, so `round` gives `10` and `int` truncates to `9` —
a different `segment_size`, and therefore a different `stored_activations`, on more than half of
all inputs.)

In roughly increasing difficulty:
[compare full-store memory against checkpointing every k layers](../tasks/rwm-activation-memory-full-store-vs-checkpoint-every-k/task.md),
[weigh memory against recompute for k in {1, 2, 4}](../tasks/rwm-memory-vs-recompute-for-checkpoint-every-k-k-in-1-2-4/task.md),
[place checkpoints to land within 5% of the true optimum](../tasks/sys-optimal-sqrt-n-checkpoint-placement/task.md),
[verify a checkpointed segment's autograd gradients against an uncheckpointed run](../tasks/rwm-checkpoint-segment-autograd-recompute-verify-grads/task.md),
and [derive the optimal recompute count with the Revolve dynamic program](../tasks/rwm-optimal-checkpoint-placement-revolve-dp/task.md),
which is the exact algorithm production checkpointing schedulers are built on.

## Common mistakes

- **Truncating the square root instead of rounding.** `int(√L)` matches `round(√L)` only when
  `L` is a perfect square; at `L = 99` it gives `segment_size = 9` where the oracle gives `10`,
  which fails an exact-match gate on more than half the integers between consecutive squares.
- **Assuming more segments means more recomputation.** The `extra forward layers` column above
  never leaves `144` across every `S` tested — recomputation is one fixed extra pass regardless
  of how finely it's split. The cost of over-segmenting is retained-memory bookkeeping, not
  extra forward work.
- **Replaying a segment without restoring its random state.** A checkpointed block that redraws
  a fresh dropout mask on the recompute pass reconstructs a *different* forward output than the
  one it originally produced, so its recomputed gradient silently disagrees with an
  uncheckpointed run — the exact failure
  [this task](../tasks/rwm-fix-a-checkpoint-that-drops-rng-state-dropout-mismatch/task.md) is
  built to catch, gated on `max_abs_err <= 1e-05` against the correct gradient.
- **Checkpointing every single layer expecting a free reduction.** At `S = 144` above, retained
  memory is `145` — six times worse than the `24` at `S = 12` — because storing a checkpoint at
  every boundary and then still replaying a one-layer "segment" pays for the same activation
  twice with no forward-work savings to show for it.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) for this track — memory and offload has
112 tasks in this bank and, per that survey, nothing found elsewhere grades a checkpointing
policy at all:

- **[PyTorch Blog — Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)**
  — the most current reading on this exact topic, straight from the PyTorch team: covers
  `torch.utils.checkpoint`, the `torch.compile` min-cut partitioner, and selective checkpointing.
  Runnable snippets, no exercise wrapper.
- **[EleutherAI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/)** — the
  most-cited derivation of activation memory with and without recomputation. Formulas to read,
  not code to run against a verdict.
- **[DeepSpeed — ZeRO / ZeRO-Offload tutorials](https://www.deepspeed.ai/tutorials/zero-offload/)**
  — shows checkpointing composed with sharded optimizer state and CPU offload in one real system,
  config-JSON only; you watch memory move rather than implement anything.
- **[PyTorch Blog — Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/)**
  — a narrated case study diagnosing a memory bug with the Memory Snapshot tool. Builds intuition
  this page's table skips past; still nothing to be graded on.

## References

1. Chen, T., Xu, B., Zhang, C., Guestrin, C., *Training Deep Nets with Sublinear Memory Cost*,
   2016 — the paper this page's `√L` segment size comes from.
   https://arxiv.org/abs/1604.06174
2. PyTorch documentation, `torch.utils.checkpoint`.
   https://docs.pytorch.org/docs/stable/checkpoint.html
3. PyTorch Blog, *Activation Checkpointing Techniques in PyTorch*, 2025-03-05.
   https://pytorch.org/blog/activation-checkpointing-techniques/
