# Concepts

One page per concept, each carrying at least one number measured by this repo's own
simulator and the exact command that regenerates it. That is the only thing these pages have
that the existing writing on these topics does not — of fifteen page-one competitors profiled
across these terms, **none** had a single interactive or reproducible element.

Every page also ends with an honest comparison to whatever else exists for that concept,
drawn from [the landscape survey](../LANDSCAPE.md). Where something out there is a better
teacher than this bank, the page says so and links it.

## Written

| Page | What it measures |
|---|---|
| [What is memory coalescing?](memory-coalescing.md) | 128-byte transactions against read stride, 2 → 33 |
| [What is warp divergence?](warp-divergence.md) | divergences and cycles against which lanes take the branch |
| [What are shared memory bank conflicts?](cuda-shared-memory-bank-conflicts.md) | conflict waves against row stride, and why stride 64 is as bad as 32 |
| [What is false sharing?](false-sharing.md) | coherence invalidations against counter padding, 7,999 → 0 |
| [What is cache blocking?](cache-blocking.md) | cache misses against tile size, including a tile that is worse than not blocking at all |
| [What is kahan summation?](kahan-summation.md) | absolute error of naive, pairwise and compensated summation against the exact sum |
| [What is log sum exp?](log-sum-exp.md) | the exact input at which the naive form overflows |
| [What is bfloat16 vs float16?](bfloat16-vs-float16.md) | range, epsilon and the first value that overflows each format |
| [What is the int8 range?](integer-quantization-ranges.md) | levels and quantization error for int4 / int8 / int16, symmetric and asymmetric |
| [What is python slots?](python-slots.md) | bytes per instance with and without `__slots__` |
| [What is kmeans?](kmeans.md) | iterations and final inertia, random init against k-means++ |
| [What is PCA?](pca.md) | explained variance and reconstruction error per component, and what skipping centering costs |

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

`python multiprocessing` · `softmax vs sigmoid` · `cache locality` · `simd` ·
`python descriptors` · `loop unrolling` · `gradient checkpointing` · `continuous batching` ·
`rmsnorm vs layernorm` · `gguf vs safetensors`

Two rules for that list. Merge synonyms onto one page — `log sum exp` and `logsumexp` are the
same page, and splitting them is how a site cannibalises itself. And drop anything whose
searchers want something else: `pose graph optimization` is robotics SLAM, `sqlite virtual
table` is not this project, however well the keyword scores.

A concept we cannot measure does not get a page. `cuda pinned memory` has no entry because
host-to-device transfers are not modelled here, and a plausible invented number is the one
failure this format cannot survive.
