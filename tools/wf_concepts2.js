export const meta = {
  name: 'concept-pages-wave-2',
  description: 'Ten more concept pages, each carrying a number measured by this repo and passing both page gates.',
  phases: [{ title: 'Write', detail: 'one agent per concept: measure, write, pass both gates' }],
}

// Every entry has a measurement that is a COUNT or an EXACT numeric property, never a
// duration. Nothing in this project is graded on a clock and no page may introduce one.
const CONCEPTS = [
  { slug: 'python-multiprocessing', term: 'python multiprocessing', track: 'Deep Python',
    idea: `The cost that decides whether multiprocessing helps is what crosses the process
boundary. Measure BYTES PICKLED for the same logical argument in different shapes — a 1M-element
list of ints vs the equivalent numpy array vs an array passed through shared memory; a generator
vs a materialised list; a closure vs a top-level function (one of which cannot be pickled at all
under the spawn start method). Use len(pickle.dumps(obj, protocol=5)) and out-of-band buffers.
Say what start method macOS and Windows default to (spawn) and why that makes the picklability
of the target function a correctness question, not a performance one.` },

  { slug: 'softmax-vs-sigmoid', term: 'softmax vs sigmoid', track: 'LLM internals',
    idea: `They are the same function when there are two classes: softmax([a,b])[1] equals
sigmoid(b-a) exactly. Show that with a measured max absolute difference over a range of inputs,
then show where they stop agreeing (more than two logits). Add the numerical half: the exact
input magnitude at which the naive sigmoid overflows in float32 vs float64, and why
softmax needs the max-shift while sigmoid needs a branch on the sign.` },

  { slug: 'cache-locality', term: 'cache locality', track: 'CPU performance',
    idea: `Walk the same 2-D array row-major and column-major, feed both address traces to
mlsys.cachesim.simulate, and count misses. Vary the row length so the column stride crosses the
line size and then the cache size. The number to name is the row length at which column-major
stops being merely worse and becomes one miss per access. Absorbs "spatial locality",
"temporal locality".` },

  { slug: 'loop-unrolling', term: 'loop unrolling', track: 'CPU performance',
    idea: `Count loop-control work, which is what unrolling removes: for an N-element loop
unrolled by factor U, count the increments, compares and branches executed, plus the remainder
iterations when U does not divide N. Table over U = 1,2,4,8,16 for a fixed N that is deliberately
NOT a multiple of the larger factors, so the tail cost is visible. The point most treatments miss
is that the tail can eat the saving: name the N and U where that happens.` },

  { slug: 'python-descriptors', term: 'python descriptors', track: 'Deep Python',
    idea: `Precedence is the whole subject and it is countable. Instrument __get__/__set__ on a
data descriptor and a non-data descriptor, put an entry of the same name in the instance
__dict__, and COUNT how many times each protocol method fires for a read and for a write. The
data descriptor wins over the instance dict; the non-data descriptor loses to it. A table of
"lookup path -> which __get__ fired, how many times" is the measurement. Also count what
__set_name__ fires on, once, at class creation.` },

  { slug: 'gradient-checkpointing', term: 'gradient checkpointing', track: 'Memory and offload',
    idea: `Pure arithmetic, exactly reproducible: for an L-layer network split into S segments,
count activations retained and forward recomputations performed during the backward pass. Table
over S for a fixed L. The optimum is at S = sqrt(L) and the memory falls as O(sqrt(L)) while the
forward work rises by one extra pass — derive both from the counts rather than asserting them,
and name the S where retained-memory stops improving enough to be worth the recompute.` },

  { slug: 'continuous-batching', term: 'continuous batching', track: 'Batching and serving',
    idea: `Count WASTED DECODE SLOTS. Take a fixed arrival trace of requests with differing output
lengths, then simulate static batching (a batch runs until its longest member finishes, so short
requests hold their slot idle) against continuous batching (a finished slot is refilled at the
next step). Count slot-steps spent on nothing in each. The ratio is the headline, and the
interesting part is how it grows with the variance of the output lengths, not their mean.` },

  { slug: 'rmsnorm-vs-layernorm', term: 'rmsnorm vs layernorm', track: 'LLM internals',
    idea: `Two measurable differences. First, parameters and operations: LayerNorm keeps gain and
bias and subtracts the mean; RMSNorm keeps only the gain. Count both exactly for a given hidden
size. Second, when they actually differ numerically: on a fixed seeded input, report max absolute
difference between the two outputs, then repeat with the input's mean forced to zero — where they
should coincide. That isolates exactly what subtracting the mean buys.` },

  { slug: 'gguf-vs-safetensors', term: 'gguf vs safetensors', track: 'Applied quantization',
    idea: `Measure bytes per weight for the same tensor in each layout: fp16 safetensors against
GGUF-style k-quant super-blocks (Q8_0, Q4_K-like) implemented as real bit-packing, plus the
per-block scale and zero-point overhead that the headline "4 bits" leaves out. Table: format,
bits per weight nominal, bytes per weight ACTUAL including block metadata, and the resulting
max absolute reconstruction error on a fixed seeded tensor. The actual-vs-nominal gap is the
measurement nobody tabulates.` },

  { slug: 'simd', term: 'simd', track: 'CPU performance',
    idea: `Count operations, not time. For an N-element elementwise loop at vector widths
W = 1,2,4,8,16, count vector operations issued, scalar tail operations, and total ops. Pick an N
that is not a multiple of the larger widths so the tail is real. Then the part that matters:
show a case where widening does NOT reduce total ops proportionally — an unaligned start, or a
loop whose trip count is smaller than the vector width — and name the N below which vectorising
cannot pay. Absorbs "simd vectorization", "simd intrinsics".` },
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
subject inside the first four words. Sentence 2 is the consequence, with a number in it.
Sentence 3 forward-references the measurement.>

## How it works

<300-450 words of plain mechanism. Inline-link 4-8 siblings. Pages that already exist and
may be linked: memory-coalescing.md, false-sharing.md, warp-divergence.md,
cuda-shared-memory-bank-conflicts.md, cache-blocking.md, kahan-summation.md, log-sum-exp.md,
bfloat16-vs-float16.md, integer-quantization-ranges.md, python-slots.md, kmeans.md, pca.md —
and any page your wave-2 siblings are writing, listed in the brief. Task statements link as
../tasks/<id>/task.md; VERIFY each directory exists with test -d first.>

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

<2-4 sentences reading the table: the pattern, and where it breaks.>

## Practise it

\`\`\`bash
mlsys grade <task-id>
\`\`\`

<Which gates, quoted from meta.json, what the shipped starter gets wrong, then 3-5 more
tasks as markdown links.>

## Common mistakes

- **<mistake>.** <why it is wrong, with the number it costs>

## Where else to practise this

<3-5 entries from LANDSCAPE.md for this track, each an honest sentence. If something there
teaches this better than this bank does, SAY SO.>

## References

1. <primary source with a URL>
2. <primary source with a URL>
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['slug', 'gate_passed', 'reproduced', 'measurement', 'tasks_linked', 'notes'],
  properties: {
    slug: { type: 'string' },
    gate_passed: { type: 'boolean', description: 'true only if BOTH check_page.py and verify_pages.py printed ok for your page' },
    reproduced: { type: 'boolean', description: 'true only if verify_pages.py confirmed your snippet reproduces the table' },
    measurement: { type: 'string', description: 'The table as plain text, exactly as in the page.' },
    tasks_linked: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string', description: 'What you changed about the suggested measurement and why, anything you could not measure, anything the reviewer should check.' },
  },
}

const SIBLINGS = CONCEPTS.map((c) => c.slug + '.md').join(', ')

const CONTEXT = `You are writing one page for github.com/mikamika06/mlsys-lab, a bank of
auto-graded exercises in low-level ML systems. Work in /Users/macbook/mlsys-lab.

=== THE ONE RULE THAT MATTERS ===
Every number in your page must come from code you actually ran, and the page must contain the
snippet that regenerates it. Never write a number you have not seen printed. If the snippet
prints something other than what you expected, the printed value is the truth and the prose
gets rewritten around it. An invented plausible number is the only unforgivable failure here:
the whole premise is that the reader can check the page, and tools/verify_pages.py will.

NO WALL-CLOCK TIMING. Nothing in this project is graded on a clock, because a duration is not
reproducible. Measure counts, exact numeric properties, byte sizes, or error magnitudes.

=== WHAT THE ENGINE CAN MEASURE ===
The package is installed; \`python3 -c "import mlsys"\` works.
  from mlsys.sim import GPU, CudaProgram
     launch(gpu, name, blocks, threads, args) -> dict with transactions, smem_waves,
     smem_insts, mem_insts, divergences, cycles, cycles_per_warp, warps, atomics, races
  from mlsys import cachesim
     cachesim.simulate(addresses, line_bytes=64, sets=64, ways=8, policy='lru')
     -> {hits, misses, evictions, accesses, miss_rate}
  from mlsys import scorers
     max_abs_err, rel_err, mse, mean_kl, argmax_agreement, byte_exact_fraction,
     channel_rel_err, size_ratio
Plain deterministic Python or numpy counting is equally valid — the false-sharing page counts
coherence invalidations with a hand-written loop, and the gradient-checkpointing style of
"count the operations a scheme performs" needs no library at all.

=== READ THESE FIRST ===
  concepts/memory-coalescing.md    the pattern, and its measured table
  concepts/false-sharing.md        hand-rolled measurement, same rigour
  concepts/cache-blocking.md       a cachesim-based measurement with an honest surprise in it
  concepts/README.md               the rules
  LANDSCAPE.md                     what else exists per track, for your last section
  tools/check_page.py              gate 1 — structure. Short; read it.
  tools/verify_pages.py            gate 2 — it RUNS your snippet. Short; read it.

=== FINDING TASKS TO LINK ===
  ls tasks | grep -i <keyword>
  python3 -c "import json;print(json.load(open('tasks/<id>/meta.json'))['gates'])"
Quote gate thresholds exactly. Every linked id must exist: check with test -d tasks/<id>.
If your concept genuinely has no matching task, say so in notes and link the closest
relatives rather than inventing an id — a broken link fails the gate anyway.

=== BOTH GATES MUST PASS ===
  python3 tools/check_page.py concepts/<slug>.md
  python3 tools/verify_pages.py concepts/<slug>.md
Iterate until both print ok. Do NOT edit either tool to make your page pass; if you do, you
have failed the task. If your page needs a package beyond numpy, put it in the page's own
\`pip install\` line — verify_pages.py reads that line and will tell you it is missing rather
than blaming the page.

=== SCOPE ===
Write exactly one file: concepts/<slug>.md. Touch nothing else. Your wave-2 siblings are
writing these concurrently, so they may not exist yet when you link them: ${SIBLINGS}. Only
link a sibling page if the file is already on disk when you check.`

phase('Write')

const results = await parallel(CONCEPTS.map((c) => () =>
  agent(
    `${CONTEXT}

=== YOUR CONCEPT ===
slug:  ${c.slug}
term:  ${c.term}      (this exact phrase is the primary keyword; use it verbatim)
track: ${c.track}

What to measure — a starting point, not a script. If a better measurement of the same idea
presents itself while you work, take it and say so in notes:
${c.idea}

=== TEMPLATE ===
${TEMPLATE.replace(/\{TERM\}/g, c.term)}

Write concepts/${c.slug}.md, make both gates print ok, and return the JSON.`,
    { label: `page:${c.slug}`, phase: 'Write', schema: SCHEMA, model: 'sonnet' }
  )
))

const ok = results.filter(Boolean)
const good = ok.filter((r) => r.gate_passed && r.reproduced)
log(`${good.length}/${CONCEPTS.length} pages passed both gates`)

return {
  passed: good.map((r) => r.slug),
  failed: ok.filter((r) => !(r.gate_passed && r.reproduced))
             .map((r) => ({ slug: r.slug, gate: r.gate_passed, reproduced: r.reproduced, notes: r.notes })),
  pages: ok,
}
