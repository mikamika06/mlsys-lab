# Concepts

One page per concept, each carrying at least one number measured by this repo's own
simulator and the exact command that regenerates it. That is the only thing these pages have
that the existing writing on these topics does not — of fifteen page-one competitors profiled
across these terms, **none** had a single interactive or reproducible element.

Every page also ends with an honest comparison to whatever else exists for that concept,
drawn from [the landscape survey](../LANDSCAPE.md). Where something out there is a better
teacher than this bank, the page says so and links it.

## Written

| Page | Concept | Measured in the page |
|---|---|---|
| [What is memory coalescing?](memory-coalescing.md) | GPU global-memory access | 128-byte transactions against read stride, 2 → 33 |
| [What is false sharing?](false-sharing.md) | CPU cache-line coherence | coherence invalidations against counter padding, 7,999 → 0 |

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
  into the 2,052 task pages, which are otherwise unreachable: GitHub's `robots.txt` blocks
  `/*/tree/`, so nothing but a hand-written link reaches them.
- **2+ references with URLs**, primary sources.
- Prose may be drafted with help; **the number, the gate thresholds, the reproduction command
  and the citations may not**. Google's own wording makes production method irrelevant and
  value the only test — a page whose only input was a keyword is the thing that gets
  penalised.

`tools/check_page.py` enforces all of it mechanically, including resolving every relative
link, because a published page pointing at a task that does not exist is worse than no page:

```bash
python3 tools/check_page.py                 # all pages
python3 tools/check_page.py docs/concepts/memory-coalescing.md
```

## Next

Ordered by how contested the term is, then by how much the measurement adds. The first ten
still to write:

`kmeans` · `python multiprocessing` · `softmax vs sigmoid` · `cache locality` · `simd` ·
`int4/int8/int16 ranges` · `python descriptors` · `pca` · `loop unrolling` ·
`gradient checkpointing`

Two rules for that list. Merge synonyms onto one page — `log sum exp` and `logsumexp` are the
same page, and splitting them is how a site cannibalises itself. And drop anything whose
searchers want something else: `pose graph optimization` is robotics SLAM, `sqlite virtual
table` is not this project, however well the keyword scores.
