---
title: "What is paged attention?"
description: "Paged attention explained, with a measured fragmentation-vs-block-table-overhead table you can reproduce without a GPU, plus a graded C++ allocator exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is paged attention?

Paged attention allocates a request's key-value cache from fixed-size
blocks pulled out of a shared pool, instead of reserving one contiguous
span sized to the longest sequence the engine will ever allow. On a
64-request mixture below, that swap drops the allocated-to-used memory
ratio from 7.33x under one worst-case contiguous span per request down to
1.03x at 16-token blocks, while multiplying the block-table entries the
engine has to track by 14.42x over that same contiguous baseline. The
table below measures that tradeoff at block sizes 1, 8, 16, 32 and 64, and
names the exact block size where growing fragmentation overtakes
shrinking table overhead.

## How it works

PagedAttention (the vLLM paper's name for the technique) borrows the
operating system's page-table trick and applies it to the KV cache.
Before it, a serving engine had to pre-allocate one contiguous span of KV
storage per request, sized to `max_seq_len` — the longest completion the
engine will ever permit — because a normal attention kernel expects a
sequence's keys and values to sit at consecutive addresses. Almost none
of that span gets used: a request that stops after 40 tokens still
reserves room for however many the model allows, the same "reserved
space nobody read or wrote" waste that [continuous batching](continuous-batching.md)
eliminates one layer up, at the request-scheduling clock instead of the
memory-allocation layer.

Paged attention breaks the contiguous assumption. KV storage is carved
into fixed-size *blocks* of `B` tokens, held in one physical pool shared
across every running request, and each sequence gets a *block table*: a
small array mapping its logical block index `0, 1, 2, ...` to whichever
physical block the allocator happened to hand it, in whatever order.
[Growing a sequence by one token](../tasks/rwa-implement-the-paged-append-slot-mapping-write-path/task.md)
that starts a new block is a single allocator call against a
[free-list-backed pool](../tasks/rwm-block-allocator-with-free-list-logical-physical-table/task.md),
not a resize of a giant buffer, and a finished request returns its
blocks to that pool immediately.

The attention kernel itself has to change to make this work: instead of
reading Q, K, V out of one flat tensor, it reads the block table first,
then [gathers the actual key/value rows through it](../tasks/llm-pagedattention-block-table-gather/task.md)
— an indirection that turns a request's logical positions into physically
scattered addresses, the same access-pattern concern
[memory coalescing](memory-coalescing.md) raises for any read that is no
longer guaranteed contiguous. Blocks are also the unit that lets requests
*share* memory instead of duplicating it:
[two block tables can point at the identical physical block](../tasks/rwb-shared-prefix-block-table-copy-on-write/task.md)
for a shared prompt prefix and copy it only once one of them writes past
it, the same copy-on-write instinct
[python's multiprocessing](python-multiprocessing.md) fork model relies on.

None of this is free, and the cost is fixed entirely by picking `B`. A
small `B` tracks a sequence's real length closely, so little space is
wasted on the tail of its last block, but every sequence then needs many
more block-table rows, which is more bookkeeping per request and more
per-block dispatch inside the kernel. A large `B` is the mirror image. It
is the same shape of decision as [cache blocking](cache-blocking.md)'s
tile-size choice and [gguf vs safetensors](gguf-vs-safetensors.md)'s
per-block scale overhead: a fixed-size chunk trading internal waste
against per-chunk metadata, and the right chunk size is something to
measure, not guess — which is what the table below does.

## Fragmentation vs block-table overhead, measured across block size

64 sequence lengths (11 to 1,637 tokens, a short-heavy distribution with a
long tail) allocated five ways: one contiguous span per sequence sized to
the batch maximum, and paged allocation at block sizes 1, 8, 16, 32 and
64. For each, the table counts allocated slots, wasted slots (internal
fragmentation — the unused tail of a sequence's last block, or of its
whole contiguous span), and block-table entries (one per block a paged
sequence holds; one per sequence for the contiguous baseline, which needs
no table at all).

| strategy | allocated slots | fragmentation (slots) | frag % of allocated | block-table entries |
|---|---|---|---|---|
| contiguous (max=1,637) | 104,768 | 90,470 | 86.35% | 64 |
| paged, b=1 | 14,298 | 0 | 0.00% | 14,298 |
| paged, b=8 | 14,512 | 214 | 1.47% | 1,814 |
| paged, b=16 | 14,768 | 470 | 3.18% | 923 |
| paged, b=32 | 15,392 | 1,094 | 7.11% | 481 |
| paged, b=64 | 16,448 | 2,150 | 13.07% | 257 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def paged_stats(lengths, b):
    lengths = np.asarray(lengths)
    blocks = -(-lengths // b)                 # ceil division per sequence
    allocated = int((blocks * b).sum())
    used = int(lengths.sum())
    frag = allocated - used
    table_entries = int(blocks.sum())
    return allocated, used, frag, table_entries

rng = np.random.default_rng(7)
N = 64
lengths = np.clip(np.round(rng.lognormal(mean=5.2, sigma=1.1, size=N)), 1, 4096).astype(int)
Lmax = int(lengths.max())
used_total = int(lengths.sum())

contig_alloc = N * Lmax
contig_frag = contig_alloc - used_total
print(f"N={N} used_total={used_total} Lmax={Lmax}")
print(f"contig alloc={contig_alloc} frag={contig_frag} frag_pct={100*contig_frag/contig_alloc:.2f} table_entries={N}")

for b in (1, 8, 16, 32, 64):
    alloc, used, frag, tbl = paged_stats(lengths, b)
    print(f"b={b} alloc={alloc} frag={frag} frag_pct={100*frag/alloc:.2f} table_entries={tbl}")

# find the exact block size, at integer resolution, where fragmentation first
# exceeds block-table entry count
prev = None
for b in range(1, 65):
    _, _, frag, tbl = paged_stats(lengths, b)
    if prev is not None:
        pf, pt = prev
        if pf <= pt and frag > tbl:
            print(f"exact_crossover prev_b={b-1} prev_frag={pf} prev_tbl={pt} b={b} frag={frag} tbl={tbl}")
    prev = (frag, tbl)
PY
```

Small blocks really do waste less: `b=1` has zero fragmentation and `b=64`
still only reaches 13.07%, both far below the contiguous baseline's
86.35%. But the block-table entry count runs the opposite direction —
`b=1` needs 14,298 entries, one per token — so the two curves cross
somewhere in between. Within the candidate set the crossing falls
between `b=16` (923 table entries vs. 470 wasted slots — table still
bigger) and `b=32` (481 vs. 1,094 — fragmentation has taken the lead); a
finer integer sweep pins the exact crossover at block size 24, where
fragmentation (702) first exceeds table entries (625) after sitting below
them at block size 23 (629 vs. 649). Every paged candidate still beats
contiguous pre-allocation outright: even the worst one here, `b=64`, wastes
90,470 ÷ 2,150 ≈ 42.08x fewer slots than reserving every sequence's slot
up front.

## Practise it

```bash
mlsys grade cpu-paged-attention-kv-block-allocator-vs-block-size
```

[That task](../tasks/cpu-paged-attention-kv-block-allocator-vs-block-size/task.md)
is real C++, compiled by `clang++`: given a fixed 30-token workload and a
per-block table-overhead cost in bytes, sweep the seven power-of-two
candidates `{16, 32, ..., 1024}` and return whichever minimizes total
allocated bytes, breaking ties toward the smaller block size. It gates on
`exact_match == 1.0` against the printed `(block_size, useful_bytes,
allocated_bytes)` triple, so a candidate that isn't the true minimizer,
or `total_blocks` computed with `floor` instead of `ceil`, changes at
least one of the three numbers and fails outright. The shipped starter
writes zero to every output.

More of the same mechanism, in roughly increasing difficulty:
[block-table gather, then attend](../tasks/rwa-implement-the-block-table-gather-then-attend/task.md),
[the paged append slot-mapping write path](../tasks/rwa-implement-the-paged-append-slot-mapping-write-path/task.md),
[max concurrent sequences that fit in the block pool](../tasks/rwm-max-concurrent-sequences-that-fit-in-the-block-pool/task.md),
[internal fragmentation, paged vs. contiguous pre-alloc](../tasks/rwb-internal-fragmentation-paged-vs-contiguous-pre-alloc/task.md)
and its sibling
[fragmentation waste fraction, paged vs. preallocated](../tasks/sys-fragmentation-waste-fraction-paged-vs-preallocated/task.md),
a debugging task,
[gather that ignores the partial last-block length](../tasks/rwa-debug-gather-ignores-the-partial-last-block-length/task.md),
and the quantized-KV version of the whole pipeline:
[paged FP8 KV — gather, dequant, attention](../tasks/rwc-paged-fp8-kv-gather-dequant-attention/task.md).

## Common mistakes

- **Rounding block count with `floor` instead of `ceil`.** A sequence of
  length 65 at `b=64` needs 2 blocks, not 1 — `floor(65/64)` silently
  drops the last token's KV entry, which the gate above catches as soon
  as `total_blocks` disagrees with the reference.
- **Forgetting the partial last block when gathering.** A block table
  entry for a sequence's final, half-full block still has to report only
  that sequence's real remaining length, not the whole block width — the
  exact bug the [partial-last-block gather task](../tasks/rwa-debug-gather-ignores-the-partial-last-block-length/task.md)
  is built around.
- **Picking block size by fragmentation alone.** `b=1` has literally zero
  fragmentation in the table above, and is still the worst real choice:
  14,298 block-table rows to maintain and dereference per decode step,
  against 257 at `b=64`.
- **Assuming a bigger block size only ever helps.** Past the crossover
  point (block size 24 here) each doubling of `b` keeps trading a
  shrinking, already-small table-overhead saving for a growing
  fragmentation cost — `b=64`'s 13.07% waste is worse than `b=32`'s
  7.11%, not better.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[tspeterkim/paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)**
  — a small, readable block-table cache manager on top of a Llama-3
  forward pass, reusing FlashAttention's PagedAttention kernel. The best
  short reference found for real block-table mechanics without reading
  the entire vLLM codebase; no tests, dormant since 2024.
- **[vLLM — PagedAttention design doc](https://docs.vllm.ai/en/latest/design/paged_attention/)**
  — vLLM's own walkthrough of the CUDA kernel: block-structured KV cache,
  per-thread-group key access, softmax and write-out. Documentation and
  source, explicitly not something you are tested against.
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** — a
  ~1,200-line vLLM reimplementation with a real `block_manager.py` doing
  paged KV-cache allocation end to end. Closest thing found to seeing the
  whole allocator in one file; read-and-run, no grading.
- **Efficiently Serving LLMs with vLLM (DeepLearning.AI)** — the newer of
  two short video courses covers PagedAttention, prefix caching and
  quantization against the real vLLM stack. Free tier is watch-and-run;
  a graded assignment sits behind the paid Pro tier.

## References

1. Kwon, W. et al., *Efficient Memory Management for Large Language Model
   Serving with PagedAttention*, SOSP 2023.
   https://arxiv.org/abs/2309.06180
2. vLLM Team, *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention*
   (announcement blog post). https://blog.vllm.ai/2023/06/20/vllm.html
3. vLLM documentation, *PagedAttention design doc*, block-table and
   physical-block-pool walkthrough.
   https://docs.vllm.ai/en/latest/design/paged_attention/
