export const meta = {
  name: 'concept-pages',
  description: 'Write one concept page per term, each carrying a number actually measured by this repo\'s engine, each passing tools/check_page.py.',
  phases: [{ title: 'Write', detail: 'one agent per concept: measure, write, pass the gate' }],
}

// Only concepts where a real number can be produced with the engine that exists.
// A page without a measurement does not ship, so a concept we cannot measure is
// not on this list — `cuda pinned memory` is absent because host transfers are
// not modelled, and inventing a plausible number is the one unforgivable failure.
const CONCEPTS = [
  { slug: 'warp-divergence', term: 'warp divergence', track: 'GPU / CUDA',
    metric: '`divergences` and `cycles` from mlsys.sim.GPU',
    idea: `Vary how threads within one warp branch — all-take, half-take, alternating lanes,
one lane only — and count divergences and cycles. The interesting result is that a branch
where every lane agrees costs nothing, and the cost depends on lane pattern, not on how
many threads take the branch.` },

  { slug: 'cuda-shared-memory-bank-conflicts', term: 'shared memory bank conflicts', track: 'GPU / CUDA',
    metric: '`smem_waves` and `smem_insts` from mlsys.sim.GPU',
    idea: `A shared-memory tile read down its columns. Vary the row stride (32, 33, 34 …) and
count the bank-conflict waves. The +1 padding trick should show up as a cliff from 32 waves
to 1. Secondary keyword: "bank conflict padding".` },

  { slug: 'cache-blocking', term: 'cache blocking', track: 'CPU performance',
    metric: '`misses` and `miss_rate` from mlsys.cachesim.simulate',
    idea: `A matrix multiply or transpose access trace, tiled at block sizes 1, 8, 16, 32, 64,
128, run through the cache model. Misses should fall and then rise again once the tile stops
fitting — naming where it turns around is the whole value. Absorbs "loop tiling", "cache
tiling".` },

  { slug: 'kahan-summation', term: 'kahan summation', track: 'Numerics',
    metric: 'absolute error against an exact reference (math.fsum or integer arithmetic)',
    idea: `Sum a sequence engineered to cancel — e.g. 1.0 followed by N copies of 1e-16 — with
naive float summation, pairwise summation, and Kahan compensated summation, against the exact
answer. Vary N. Kahan should hold error near zero where naive loses everything.` },

  { slug: 'log-sum-exp', term: 'log sum exp', track: 'Numerics',
    metric: 'overflow/underflow and absolute error vs the shifted form',
    idea: `Compute log(sum(exp(x))) naively and with the max-shift trick for inputs of growing
magnitude (10, 100, 710, 1000). Naive overflows to inf at a specific value — find and state it
exactly. Also show the underflow side with large negative inputs. Absorbs "logsumexp".` },

  { slug: 'bfloat16-vs-float16', term: 'bfloat16 vs float16', track: 'Numerics',
    metric: 'exact representable range and the first value that overflows each format',
    idea: `Both are 16 bits; bf16 spends more on exponent and less on mantissa. Produce a table:
exponent/mantissa bits, max finite value, smallest normal, epsilon, and the exact value at
which each overflows to inf. Use numpy dtypes (np.float16, and ml_dtypes or a manual bf16
round-trip via float32 truncation — state which you used). Absorbs "fp16 vs fp32".` },

  { slug: 'integer-quantization-ranges', term: 'int8 range', track: 'Quantization',
    metric: 'representable range and maximum quantization error per bit width',
    idea: `Symmetric and asymmetric integer ranges for int4/int8/int16: min, max, number of
levels, and — the part nobody tabulates — the worst-case and mean absolute quantization error
when a real tensor is mapped into each. Compute it on a fixed seeded normal tensor. Absorbs
"int4 range", "int16 range", "int4 max value".` },

  { slug: 'python-slots', term: 'python slots', track: 'Deep Python',
    metric: 'bytes per instance, measured',
    idea: `Instances of a plain class vs one with __slots__: measure real per-instance cost.
sys.getsizeof alone is misleading because it excludes the __dict__, so measure the total —
getsizeof(obj) + getsizeof(obj.__dict__) where it exists — and cross-check against a bulk
allocation with tracemalloc for N instances. State both numbers and why they differ. Also
show what __slots__ takes away (no new attributes, no weakref by default).` },

  { slug: 'kmeans', term: 'kmeans', track: 'Algorithms',
    metric: 'iterations to convergence and final inertia, seeded',
    idea: `Lloyd's algorithm on a fixed seeded dataset. Compare random init against k-means++
init: iterations to convergence and final inertia, over several seeds, as a table. k-means++
should converge in fewer iterations and land in a better local minimum more often — report the
actual spread, not just the mean, and say plainly if the advantage is small.` },

  { slug: 'pca', term: 'pca', track: 'Algorithms',
    metric: 'explained variance ratio and reconstruction error by component count',
    idea: `PCA on a fixed seeded dataset via the covariance eigendecomposition and via SVD:
show they agree, then a table of cumulative explained variance and reconstruction error for
k = 1..n components. Include the detail that trips people up — centering — by showing what the
numbers become when you skip it.` },
]

const TEMPLATE = `Front matter, then this shape. Deviating breaks the gate.

---
title: "What is {TERM}?"
description: "<{TERM} explained, with a measured <metric> you can reproduce, plus a graded exercise.>"
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is {TERM}?

<FIRST 50 WORDS. Sentence 1 is a standalone definition with {TERM} as the grammatical
subject inside the first four words — this is what an AI Overview will quote. Sentence 2
is the consequence, with a number in it. Sentence 3 forward-references the measurement.>

## How it works

<300-450 words of plain mechanism. Inline-link 4-8 sibling concepts here. Existing pages you
may link: concepts/memory-coalescing.md, concepts/false-sharing.md. You may also link task
statements as ../tasks/<id>/task.md — VERIFY each directory exists first.>

## <a heading naming the measurement>

<1-3 sentences on what was varied and what was counted.>

| <varied input> | <metric> | <second metric> |
|---|---|---|
| ... | ... | ... |

Reproduce it:

\`\`\`bash
pip install mlsys-lab
python3 - <<'PY'
<the exact 15-30 lines that produced the table>
PY
\`\`\`

<2-4 sentences reading the table: the pattern, and where it breaks. Naming the place it
breaks is what separates this from a definition.>

## Practise it

\`\`\`bash
mlsys grade <task-id>
\`\`\`

<Which gates, what the shipped starter gets wrong, then 3-5 more tasks as markdown links.>

## Common mistakes

- **<mistake>.** <why it is wrong, with the number it costs>

## Where else to practise this

<3-5 entries taken from LANDSCAPE.md for this track, each an honest sentence. If something
there is better than this bank at teaching this, SAY SO — that honesty is the point.>

## References

1. <primary source with a URL>
2. <primary source with a URL>
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['slug', 'gate_passed', 'measurement', 'reproduced', 'tasks_linked', 'notes'],
  properties: {
    slug: { type: 'string' },
    gate_passed: { type: 'boolean', description: 'true only if tools/check_page.py printed ok for your page' },
    reproduced: { type: 'boolean', description: 'true only if you RAN the snippet in the page and its output matches the table verbatim' },
    measurement: { type: 'string', description: 'The table as plain text, exactly as it appears in the page.' },
    tasks_linked: { type: 'array', items: { type: 'string' }, description: 'Task ids you linked. Every one must exist under tasks/.' },
    notes: { type: 'string', description: 'Anything the reviewer must know: a claim you could not measure, a metric that behaved unexpectedly, a place you were unsure.' },
  },
}

const CONTEXT = `You are writing one page for github.com/mikamika06/mlsys-lab, a bank of 2053
auto-graded exercises in low-level ML systems. Work in /Users/macbook/mlsys-lab.

=== THE ONE RULE THAT MATTERS ===
Every number in your page must come from code you actually ran, and the page must contain the
snippet that regenerates it. Do not write a number you have not seen printed. If your snippet
prints something different from what you expected, the printed value is the truth and the
prose must be rewritten around it. A plausible invented number is the only unforgivable
failure here, because the entire premise of these pages is that the reader can check them.

=== WHAT THE ENGINE CAN MEASURE ===
Import from the installed package (it is installed: \`python3 -c "import mlsys"\` works).
  from mlsys.sim import GPU, CudaProgram
     launch(gpu, name, blocks, threads, args) returns a dict with:
     transactions, smem_waves, smem_insts, mem_insts, divergences, cycles,
     cycles_per_warp, warps, atomics, races, result
  from mlsys import cachesim
     cachesim.simulate(addresses, line_bytes=64, sets=64, ways=8, policy='lru')
     -> {hits, misses, evictions, accesses, miss_rate}
  from mlsys import scorers
     max_abs_err, rel_err, mse, mean_kl, argmax_agreement, byte_exact_fraction,
     channel_rel_err, size_ratio
Plain deterministic Python or numpy counting is also fine — the false-sharing page counts
coherence invalidations with a hand-written loop. What is NOT fine is wall-clock timing:
nothing in this project is graded on a clock, and a page must not introduce one.

=== READ THESE FIRST ===
  concepts/memory-coalescing.md   the pattern to follow, and its measured table
  concepts/false-sharing.md       the second pattern, hand-rolled measurement
  concepts/README.md              the rules
  LANDSCAPE.md                    what else exists per track, for your last section
  tools/check_page.py             the gate you must pass — read it, it is short

=== FINDING THE TASKS TO LINK ===
The bank is in tasks/. Find yours with, for example:
  ls tasks | grep -i divergen
  python3 -c "import json;print(json.load(open('tasks/<id>/meta.json'))['gates'])"
Quote gate thresholds from meta.json exactly. Every task id you link MUST exist — check with
\`test -d tasks/<id>\`. A page that links a task that does not exist fails the gate anyway.

=== THE GATE ===
  python3 tools/check_page.py concepts/<slug>.md
It requires front matter with title/description/datePublished/dateModified, 600-1400 prose
words, a markdown table, a "Reproduce it" section, a runnable fenced block, a "mlsys grade"
line placed AFTER the first table, 8-25 relative links that all resolve, and a "## References"
section with 2+ numbered items containing URLs. Iterate until it prints ok. Do not weaken the
gate to pass it — if you edit tools/check_page.py, you have failed the task.

=== SCOPE ===
Write exactly one file: concepts/<slug>.md. Touch nothing else in the repo.`

phase('Write')

const results = await parallel(CONCEPTS.map((c) => () =>
  agent(
    `${CONTEXT}

=== YOUR CONCEPT ===
slug:   ${c.slug}
term:   ${c.term}       (this exact phrase is the primary keyword; use it verbatim)
track:  ${c.track}
metric: ${c.metric}

What to measure — a starting point, not a script. If a better measurement of the same idea
presents itself while you work, take it and say so in notes:
${c.idea}

=== TEMPLATE ===
${TEMPLATE.replace(/\{TERM\}/g, c.term)}

Write concepts/${c.slug}.md, make the gate print ok, and return the JSON.`,
    { label: `page:${c.slug}`, phase: 'Write', schema: SCHEMA, model: 'sonnet' }
  )
))

const ok = results.filter(Boolean)
const passed = ok.filter((r) => r.gate_passed && r.reproduced)
log(`${passed.length}/${CONCEPTS.length} pages passed the gate with a reproduced measurement`)

return {
  passed: passed.map((r) => r.slug),
  failed: ok.filter((r) => !(r.gate_passed && r.reproduced))
             .map((r) => ({ slug: r.slug, gate: r.gate_passed, reproduced: r.reproduced, notes: r.notes })),
  pages: ok,
}
