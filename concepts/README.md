# Concepts

One page per concept, each carrying at least one number measured by this repo's own
simulator and the exact command that regenerates it. That is the only thing these pages have
that the existing writing on these topics does not — of fifteen page-one competitors profiled
across these terms, **none** had a single interactive or reproducible element.

Every page also ends with an honest comparison to whatever else exists for that concept,
drawn from [the landscape survey](../LANDSCAPE.md). Where something out there is a better
teacher than this bank, the page says so and links it.

## Written

37 pages, each carrying a number this repo measured and the command that regenerates it.

**GPU and CUDA**

- [What is memory coalescing?](memory-coalescing.md)
- [What is warp divergence?](warp-divergence.md)
- [What are shared memory bank conflicts?](cuda-shared-memory-bank-conflicts.md)
- [What is kernel fusion?](kernel-fusion.md)
- [What are CUDA graphs?](cuda-graphs.md)

**CPU performance**

- [What is false sharing?](false-sharing.md)
- [What is cache blocking?](cache-blocking.md)
- [What is cache locality?](cache-locality.md)
- [What is loop unrolling?](loop-unrolling.md)
- [What is SIMD?](simd.md)

**C++**

- [What is move semantics?](move-semantics.md)
- [What is a virtual function table?](virtual-function-table.md)

**Numerics**

- [What is kahan summation?](kahan-summation.md)
- [What is log sum exp?](log-sum-exp.md)
- [What is bfloat16 vs float16?](bfloat16-vs-float16.md)
- [What is floating point precision in Python?](floating-point-precision-python.md)
- [What is the int8 range?](integer-quantization-ranges.md)
- [GGUF vs safetensors: what is the difference?](gguf-vs-safetensors.md)

**Attention and the KV cache**

- [What is grouped query attention (GQA)?](grouped-query-attention.md)
- [What is paged attention?](paged-attention.md)
- [What is a KV cache?](kv-cache.md)
- [What are rope embeddings?](rope-embeddings.md)
- [What is self-attention vs cross-attention?](self-attention-vs-cross-attention.md)

**LLM internals and systems**

- [What is the softmax function?](softmax-function.md)
- [What is softmax vs sigmoid?](softmax-vs-sigmoid.md)
- [What is layer normalization, and how does RMSNorm differ?](rmsnorm-vs-layernorm.md)
- [What is LLM inference?](llm-inference.md)
- [What is LLM inference optimization?](llm-inference-optimization.md)
- [What is continuous batching?](continuous-batching.md)
- [What is gradient checkpointing?](gradient-checkpointing.md)

**Compile and export**

- [What is torch compile?](torch-compile.md)
- [What is torch onnx export?](torch-onnx-export.md)

**Deep Python**

- [What is __slots__ in Python?](python-slots.md)
- [What are python descriptors?](python-descriptors.md)
- [What is python multiprocessing?](python-multiprocessing.md)

**Algorithms**

- [What is k-means clustering?](kmeans.md)
- [PCA in Python](pca.md)


## The rules these pages follow

Derived from measuring the pages that currently rank for these terms, not from folklore:

- **The filename is the keyword.** It is the only title lever a markdown page on github.com
  has — the rendered `<title>` is the file path.
- **600–1,400 prose words.** The median page-one competitor is 1,081; two of the #1 results
  measured were under 1,000. Length is not the lever, so do not pad.
- **At least one measured table, plus the command that reproduces it.** No number, no page.
- **The exercise link comes after the first table**, never in the intro, where it reads as an
  advertisement.
- **8–25 internal links**, anchor text = the thing being linked. These are the crawl path
  into the task pages, which are otherwise unreachable: GitHub's `robots.txt` blocks
  `/*/tree/`, so nothing but a hand-written link reaches them.
- **2+ references with URLs**, primary sources.
- Prose may be drafted with help; **the number, the gate thresholds, the reproduction command
  and the citations may not**. Google's own wording makes production method irrelevant and
  value the only test — a page whose only input was a keyword is the thing that gets
  penalised.

Two checks enforce all of it, and CI runs both:

```bash
python3 tools/check_page.py     # structure, word count, and every relative link resolves
python3 tools/verify_pages.py   # runs each page's own snippet and compares it to the table
```

`verify_pages.py` is the one that matters. It executes the `python3 - <<'PY'` block from each
page and requires that every number the script prints appears in the page — allowing for
rounding, since tables round, but not for a different number. A page that invites the reader
to check it and then does not survive being checked would be worse than a page with no numbers
in it at all.

## Next

Ordered by how contested the term is, then by how much a measurement adds:

`python gil` · `virtual function table` · `rvalue reference` · `torch compile` ·
`paged attention` · `knowledge distillation` · `speculative decoding` · `cuda graphs` ·
`floating point precision python` · `torch onnx export`

Two rules for that list. Merge synonyms onto one page — `log sum exp` and `logsumexp` are the
same page, and splitting them is how a site cannibalises itself. And drop anything whose
searchers want something else: `pose graph optimization` is robotics SLAM, `sqlite virtual
table` is not this project, however well the keyword scores.

A concept we cannot measure does not get a page. `cuda pinned memory` has no entry because
host-to-device transfers are not modelled here, and a plausible invented number is the one
failure this format cannot survive.
