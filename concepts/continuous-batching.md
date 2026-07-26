---
title: "What is continuous batching?"
description: "Continuous batching explained, with a measured wasted-decode-slot table you can reproduce without a GPU, plus a graded scheduler exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is continuous batching?

Continuous batching is a decode scheduler that refills a finished request's
slot with a waiting request at the very next iteration, instead of holding
that slot idle until every other member of its batch also finishes. On a
64-request trace below, that difference alone is a 1.48x to 2.74x gap in
wasted decode slot-steps between the two schedulers. The rest of this page
measures that gap and how it grows as output lengths spread out.

## How it works

The scheduler it replaces is **static (request-level) batching**: group up
to $B$ requests, decode the whole batch one token per step, and start the
next batch only once every member of the current one has produced its last
token. Every request in a batch shares one iteration clock, so a request
that needs 5 tokens sits in its slot doing nothing for however many extra
steps the slowest member of that same batch needs — a real cost, not a
rounding error, and the exact thing
[the makespan of static batching](../tasks/rwb-static-request-level-batching-makespan/task.md)
and
[its step-count baseline](../tasks/sys-static-batch-step-count-baseline/task.md)
ask you to compute.

Continuous batching (also called in-flight or iteration-level batching)
removes the shared clock. The scheduler looks at slot occupancy at every
step: any slot whose occupant just emitted its last token is freed and
immediately handed to the next waiting request, which starts contributing a
real token on the very next iteration rather than the next batch boundary.
[Join-on-finish scheduling](../tasks/sys-join-on-finish-continuous-batching/task.md)
is exactly this admit-decode-retire loop. `llama.cpp`'s `--parallel N`
server sits between the two: it keeps $N$ *physical* slots like continuous
batching, refilling one the moment it frees, but never splits or preempts a
running request mid-generation the way paged continuous batching can — see
[the fixed-slot `-np N` scheduler](../tasks/rwb-fixed-slot-np-n-batched-decode-scheduler/task.md).

The same shape of bug shows up in every real implementation: a slot that is
freed but not actually handed back leaves capacity idle forever, which is
what
[a scheduler that never frees finished slots](../tasks/rwb-debug-a-scheduler-that-never-frees-finished-slots/task.md)
asks you to catch, and admitting one request too many past the concurrency
cap is the opposite failure.

This is one instance of a pattern that recurs whenever several logical
items share one fixed-size physical unit and that unit only advances when
its slowest occupant does: a warp's 32 lanes issuing one instruction
together ([warp divergence](warp-divergence.md)), 32 threads sharing one
128-byte transaction ([memory coalescing](memory-coalescing.md)), one cache
line shared by counters that never intended to share anything
([false sharing](false-sharing.md)), and one bank serving whichever thread
gets there first
([shared-memory bank conflicts](cuda-shared-memory-bank-conflicts.md)). A
decode slot is the same idea at the request-scheduling layer, and it is
countable the same way: not in wall-clock time, but in how many
slot-iterations were spent advancing nothing.

Continuous batching does not erase the cost, it just shrinks it to the
unavoidable tail: once the arrival queue is exhausted, slots empty out one
at a time as the last few requests finish, and for that final stretch fewer
than $S$ slots are occupied no matter how good the scheduler is.

## Wasted decode slots measured against output-length variance

64 requests, 16 slots, all arriving in one burst so the only thing that
changes between rows is how spread out the output lengths are — the mean is
pinned at 30 tokens throughout. `wasted` counts slot-steps where an
allocated slot advanced no request: for static batching, every step a short
request idles inside a batch waiting on the longest member; for continuous
batching, only the empty-queue tail.

| length variance | wasted slot-steps (static) | wasted slot-steps (continuous) | ratio |
|---|---|---|---|
| 0.8 | 98 | 66 | 1.48 |
| 3.6 | 192 | 128 | 1.50 |
| 13.5 | 398 | 254 | 1.57 |
| 53.8 | 788 | 404 | 1.95 |
| 121.0 | 1,199 | 511 | 2.35 |
| 203.0 | 1,575 | 631 | 2.50 |
| 313.2 | 1,965 | 717 | **2.74** |

Reproduce it — pure counting, so the numbers are identical on every machine:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np
from collections import deque

def continuous_stats(lengths, S):
    q = deque(int(x) for x in lengths)
    active = []
    while len(active) < S and q:
        active.append(q.popleft())
    iters = used = 0
    while active:
        iters += 1
        nxt = []
        for rem in active:
            rem -= 1
            used += 1
            if rem > 0:
                nxt.append(rem)
        active = nxt
        while len(active) < S and q:
            active.append(q.popleft())
    return S * iters, used

def static_stats(lengths, S):
    alloc = used = 0
    for i in range(0, len(lengths), S):
        group = lengths[i:i + S]
        alloc += max(group) * len(group)
        used += sum(group)
    return alloc, used

N, S, MEAN = 64, 16, 30
rng = np.random.default_rng(0)
base = rng.standard_normal(N)
base -= base.mean()
for scale in (1, 2, 4, 8, 12, 16, 20):
    lengths = np.clip(np.round(MEAN + scale * base), 1, None).astype(int)
    alloc_s, used_s = static_stats(lengths, S)
    alloc_c, used_c = continuous_stats(lengths, S)
    wasted_s, wasted_c = alloc_s - used_s, alloc_c - used_c
    print(f"var={lengths.var():.1f}  wasted_static={wasted_s}  "
          f"wasted_cont={wasted_c}  ratio={wasted_s/wasted_c:.2f}")
PY
```

The headline is not that continuous batching wins — that was never in
doubt — it is that **the margin is not fixed**. At variance 0.8, every
request needs close to the same 30 tokens, so a batch's longest member
barely exceeds its shortest and static batching's blocking cost is small: a
1.48x ratio. By variance 313.2 a handful of long outliers each drag an
entire batch of 16 short requests along for the ride, and the ratio climbs
to 2.74x — over three times the static waste for barely a third more
variance in the underlying lengths. Real serving traffic (chat replies
against one-word confirmations) sits well past the high-variance end of
this table, which is why continuous batching is the default in every
production engine and not a niche optimisation.

## Practise it

```bash
mlsys grade rwb-simulate-continuous-batching-active-set-under-max-num-seqs
```

[That task](../tasks/rwb-simulate-continuous-batching-active-set-under-max-num-seqs/task.md)
gates on `exact_match == 1.0` against the full per-iteration active-set
sequence, not just a final count — FIFO admission order, a same-step
arrival tie broken by index, and an idle iteration emitted while waiting
for a late arrival all have to match exactly. The shipped starter is
`raise NotImplementedError`, so it fails immediately; the interesting way
to fail it once you have written something is to decode a request the same
step it is admitted, which desyncs every active-set list by one iteration
from there on.

In roughly increasing difficulty:
[padding waste of static batching](../tasks/rwb-measure-padding-waste-of-static-batching/task.md)
(no scheduling loop, just the formula),
[join-on-finish scheduling](../tasks/sys-join-on-finish-continuous-batching/task.md),
[the fixed-slot `-np N` scheduler](../tasks/rwb-fixed-slot-np-n-batched-decode-scheduler/task.md),
and two debugging tasks:
[a scheduler that never frees finished slots](../tasks/rwb-debug-a-scheduler-that-never-frees-finished-slots/task.md)
and
[a scheduler that overflows its slot count](../tasks/rwb-debug-a-scheduler-that-overflows-np-slots/task.md).

## Common mistakes

- **Decoding before admitting.** A request that arrives at step $t$ must
  still get its first token at step $t$ if a slot is free. Admitting it
  only in time for step $t+1$ looks like a one-step rounding difference and
  fails `exact_match` on every iteration after it, because the whole
  active-set sequence shifts.
- **Freeing a slot without refilling it the same step.** A slot marked free
  but not immediately handed to the next waiting request behaves like
  static batching for one extra iteration per occurrence — the exact bug
  [the never-frees task](../tasks/rwb-debug-a-scheduler-that-never-frees-finished-slots/task.md)
  is built around.
- **Measuring the win only at the mean.** A benchmark run at one fixed,
  narrow output-length distribution can show a 1.5x gap and call it done;
  the table above shows that number is a function of variance, not a
  constant, and the same workload with heavier-tailed lengths shows 2.74x.
- **Assuming continuous batching has zero waste.** It still pays the
  empty-queue tail — `wasted_cont` in the table is never 0. What it removes
  is the *blocking* term, not all idling.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** — a from-scratch vLLM
  reimplementation in about 1,200 readable lines, with a real `scheduler.py` and
  `block_manager.py` doing continuous batching and paged KV-cache management. The
  closest thing on GitHub to seeing the whole mechanism in one file; no test suite, so
  you read and run `bench.py` rather than get a scored answer.
- **Efficiently Serving LLMs (DeepLearning.AI / Predibase)** — a 2h40m video course
  covering KV caching, continuous batching, and multi-LoRA serving with seven
  run-along notebooks. Free tier is watch-and-run-the-cell; a graded assignment exists
  only behind the paid tier.
- **[achi9629/llm-inference-engine](https://github.com/achi9629/llm-inference-engine)**
  — a solo project building an inference engine in explicit stages (plain forward
  pass, KV cache, static batching, continuous batching, paged KV cache) with 122
  pytest tests per stage. The closest structural match to "implement the mechanic,
  then check yourself" found for this topic, but a brand-new, one-star repo rather
  than a vetted resource.
- **[Vidur](https://github.com/microsoft/vidur)** — Microsoft Research's LLM-serving
  simulator (MLSys 2024): configure a model, hardware and scheduling policy, replay a
  workload trace, and read back TTFT/TPOT/batch-size numbers. A genuine research tool
  for scheduling tradeoffs, with no notion of a correct answer to grade against — the
  opposite failure mode from a page with no measurement at all.

## References

1. Yu, G. et al., *Orca: A Distributed Serving System for Transformer-Based Generative
   Models*, OSDI 2022 — the paper that introduced iteration-level (continuous) batching.
   https://www.usenix.org/conference/osdi22/presentation/yu
2. Kwon, W. et al., *Efficient Memory Management for Large Language Model Serving with
   PagedAttention*, SOSP 2023 — vLLM, continuous batching paired with paged KV-cache
   allocation. https://arxiv.org/abs/2309.06180
3. `llama.cpp` server documentation, `--parallel` / slot-based decoding.
   https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
