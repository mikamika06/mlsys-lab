---
title: "What is false sharing?"
description: "False sharing explained, with a measured table of cache-line invalidations against counter padding that you can reproduce on any machine, plus graded C++ exercises."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is false sharing?

False sharing is the slowdown that happens when two threads write to *different* variables
that live inside the same cache line. Nothing is actually shared, so no lock is needed and no
result is wrong — but the coherence protocol works on whole lines, so each write invalidates
the line in the other core's cache and the two threads take turns owning memory neither of
them shares. Below, the cost measured as a count you can regenerate.

## How it works

A cache line is the smallest unit a cache tracks — 64 bytes on x86-64 and on Apple silicon's
L1 for these purposes. Coherence is also tracked per line: when core A writes to a line, every
other core holding that line must drop its copy before the write can complete. That rule is
what makes multi-threaded code correct, and it is indifferent to *which bytes* inside the line
you touched.

So put eight per-thread counters in an `int64_t counters[8]` array and you have put all eight
in one 64-byte line. Thread 3 increments `counters[3]`; thread 5's copy of the line is
invalidated even though thread 5 only ever reads and writes `counters[5]`. Each of the eight
threads keeps stealing the line back. The arithmetic is embarrassingly parallel, the data is
disjoint, and the program still serialises on the memory system.

The tell is that the slowdown scales with *thread count* and disappears when you add padding
that changes nothing about the logic. That is what separates it from true sharing — where
threads really do contend for the same variable and the fix is algorithmic — and from
[memory coalescing](memory-coalescing.md), which is the same cache-line granularity biting on
a GPU, where the unit is a 128-byte transaction and the cost is bandwidth rather than
coherence traffic.

Three things make it hard to spot in review. It is invisible in the source: `counters[3]` and
`counters[5]` look independent because they are. It is invisible in single-threaded profiling,
because with one thread there is nothing to invalidate. And the fix — padding each counter to
its own line, or giving each thread a local accumulator and merging once at the end — looks
like waste to anyone who has not measured it.

## Invalidations measured against counter spacing

Eight threads, each incrementing its own counter 1,000 times, interleaved round-robin. The
only thing that varies is how far apart the counters sit. An *invalidation* is counted
whenever a line is written by a different thread than the one that wrote it last — the event
the coherence protocol has to pay for.

| counter stride | counters per line | lines used | invalidations |
|---|---|---|---|
| 8 B | 8 | 1 | **7,999** |
| 16 B | 4 | 2 | 7,998 |
| 32 B | 2 | 4 | 7,996 |
| 64 B | 1 | 8 | **0** |
| 128 B | 1 | 8 | 0 |

Reproduce it — pure counting, no timing, so the numbers are the same everywhere:

```python
LINE, THREADS, ITERS = 64, 8, 1000

def trace(stride):                     # round-robin is what exposes the sharing
    return [(t * stride, t) for _ in range(ITERS) for t in range(THREADS)]

def invalidations(tr, line=LINE):
    owner, n = {}, 0
    for addr, tid in tr:
        ln = addr // line
        if owner.get(ln, tid) != tid:
            n += 1
        owner[ln] = tid
    return n

for stride in (8, 16, 32, 64, 128):
    tr = trace(stride)
    print(stride, len({a // LINE for a, _ in tr}), invalidations(tr))
```

Two things in that table are worth more than the headline. First, **padding partway does
almost nothing**: going from 8 to 32 bytes of spacing removes 3 invalidations out of 7,999,
because as long as any two threads share a line they keep trading it. The cost does not taper
off — it falls off a cliff at exactly the point where each thread owns a whole line. Second,
**128-byte stride buys nothing over 64** in this model, so padding to two lines is wasted
memory. On some hardware it does help, because adjacent-line prefetching pulls the neighbour
in as well, which is why `std::hardware_destructive_interference_size` exists rather than a
hardcoded 64.

The real cost per invalidation is tens to hundreds of cycles of coherence traffic, so ~8,000
of them is the difference between a loop that scales with cores and one that does not.

## Practise it

```bash
mlsys grade cpu-pad-per-thread-counters-to-kill-false-sharing
```

[That task](../tasks/cpu-pad-per-thread-counters-to-kill-false-sharing/task.md) is real
C++, compiled by your local `clang++`, and it gates on `exact_match == 1.0` — the padded
layout has to produce byte-identical output to the reference, so a layout that merely
*looks* padded does not pass.

Then, in rough order:
[predict which of 5 layouts false-share](../tasks/cpu-predict-which-of-5-layouts-false-share/task.md) (no code),
[classify 10 access patterns](../tasks/cpp-false-sharing-classifier-10-access-patterns/task.md),
[fix it by 64-byte padding](../tasks/cpp-fix-false-sharing-by-64b-padding/task.md), and
[design a reduction tree that minimises it](../tasks/cpu-reduction-tree-layout-minimizing-false-sharing/task.md).
The cache model they are graded against is
[`src/mlsys/sim/cache.py`](../src/mlsys/sim/cache.py), re-exported as `mlsys.cachesim`; it
is deterministic, so your score does not depend on what else your machine was doing.

## Common mistakes

- **Padding the array instead of the elements.** Adding 64 bytes at the end of
  `counters[8]` changes nothing; the eight counters are still adjacent. The padding has to sit
  *between* them.
- **Assuming `alignas(64)` on the struct is enough.** It aligns the start. If the struct is
  smaller than a line, two consecutive instances still share one.
- **Fixing it with a mutex.** A lock makes the invalidation traffic worse and adds
  contention on top, while the underlying problem — disjoint data in one line — is untouched.
- **Reading a wall-clock speedup as proof.** A padded version can measure faster for reasons
  that have nothing to do with sharing (allocator luck, frequency scaling). Counting the
  invalidations tells you *why*, which is the whole reason the gates here count instead of
  timing.
- **Padding everything, forever.** Padding costs cache capacity. It pays for
  frequently-written per-thread state and is waste on read-mostly data, which is never
  invalidated in the first place.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) — this area has real competition and
one resource that is better than this bank at explaining it:

- **[perf-ninja](https://github.com/dendibakh/perf-ninja)** — Denis Bakhvalov's lab course,
  and the closest topic-for-topic match to this whole track. Its false-sharing lab is
  excellent and it teaches with real perf counters. The difference is what it grades: a
  wall-clock speedup threshold measured on CI hardware, so the verdict depends on the machine
  by construction. Do both; it is the better teacher, this is the reproducible scorer.
- **[CS:APP Cache Lab](https://csapp.cs.cmu.edu/3e/labs.html)** — the only other resource
  found anywhere that grades on a deterministic simulated count rather than a clock. One lab,
  unchanged since about 2015, and still worth doing.
- **[Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)**
  — Igor Ostrovsky, 2010. Seven short runnable examples; example 6 is a 15x false-sharing
  slowdown. The canonical intuition-builder.
- **[Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)** — Sergey Slotin's
  free HPC book covering the same ground as reading, with no exercises. Note the name
  collision: unrelated to the Springer journal *Algorithmica*.

## References

1. Intel, *Avoiding and Identifying False Sharing Among Threads*.
   https://www.intel.com/content/www/us/en/developer/articles/technical/avoiding-and-identifying-false-sharing-among-threads.html
2. Drepper, U., *What Every Programmer Should Know About Memory*, 2007, §3.3.4 on
   multi-processor cache coherence.
   https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
3. cppreference, `std::hardware_destructive_interference_size` — why the padding constant is
   a query and not the literal 64.
   https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
