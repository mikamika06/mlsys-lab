export const meta = {
  name: 'concept-pages-wave-3',
  description: 'Ten pages targeting the measured demand the first 22 left uncovered, KD 18 and below.',
  phases: [{ title: 'Write', detail: 'one agent per concept: measure, write, pass both gates' }],
}

// Chosen from the measured gap (KEYWORD-GAP.md): highest volume that is both winnable at this
// authority (KD <= 18) and actually measurable here. The attention/KV cluster leads because
// the landscape survey found nobody anywhere grading KV-cache layout.
const CONCEPTS = [
  { slug: 'grouped-query-attention', term: 'grouped query attention', vol: '720 + 260', kd: '10-15',
    track: 'Attention / KV', also: ['multi query attention (260/mo) — absorb, do not split'],
    idea: `KV-cache bytes is the whole reason GQA exists. For a fixed model shape (say 32 query
heads, head_dim 128, 32 layers, fp16) table the KV bytes per token against the number of KV
groups: 32 groups is MHA, 1 group is MQA, 8 is the usual GQA. Show bytes per token and total for
a realistic context length. The number people never see is how quickly that dominates: at long
context the cache outweighs the weights. Compute both and say where they cross.` },

  { slug: 'paged-attention', term: 'paged attention', vol: 590, kd: 10,
    track: 'Attention / KV', also: [],
    idea: `Paged attention exists to stop wasting KV memory on requests whose length you did not
know in advance. Measure the waste: for a fixed set of sequence lengths, compare contiguous
pre-allocation to the longest possible length against block allocation at block sizes 1, 8, 16,
32, 64. Count allocated slots, used slots, and internal fragmentation (the tail of the last
block). Small blocks waste less and cost more block-table entries — table both, and name the
block size where the two curves cross.` },

  { slug: 'kv-cache', term: 'kv cache', vol: 390, kd: 9,
    track: 'Attention / KV', also: ['kv cache explained (390/mo)'],
    idea: `Bytes per token, exactly. Table KV-cache size against model shape: layers, heads,
head_dim, dtype. Then the part that decides deployments — the same table as a fraction of total
memory once weights are counted, at several context lengths, so the reader sees the context
length at which the cache exceeds the weights. Also count what recomputing instead of caching
would cost in attention operations, since that is the actual trade.` },

  { slug: 'rope-embeddings', term: 'rope embeddings', vol: 390, kd: 15,
    track: 'LLM internals', also: ['rotary position embedding'],
    idea: `RoPE's defining property is exact and checkable: the dot product of two rotated
vectors depends only on the DIFFERENCE of their positions. Verify it numerically — rotate a
fixed query and key at absolute positions (m, n) and at (m+k, n+k) and show the dot products
agree to float64 rounding, across several k. Then show what breaks with naive absolute position
embeddings, and measure how the property degrades past the trained context length, which is what
position-interpolation scaling exists to fix.` },

  { slug: 'floating-point-precision-python', term: 'floating point precision python', vol: 320, kd: 2,
    track: 'Numerics', also: [],
    idea: `Start from 0.1 + 0.2 != 0.3 but do not stop there, because every other page on this
query stops there. Show the exact stored value of 0.1 (Fraction or Decimal from the float gives
the true binary value), the gap between representable neighbours across magnitudes, the point
where adding 1.0 to a float changes nothing at all, and what Decimal and Fraction cost and fix.
Table it. The last one — the largest float where x + 1 == x — is a concrete number almost nobody
quotes.` },

  { slug: 'llm-inference', term: 'llm inference', vol: 260, kd: 0,
    track: 'LLM systems', also: ['what is llm inference (260/mo)'],
    idea: `Prefill and decode are different computations and the arithmetic shows it. For a fixed
model shape, count FLOPs and bytes moved per token for a prefill of length P against one decode
step, and derive arithmetic intensity for each. Prefill is compute-bound, decode is
memory-bound, and the ratio is what every serving decision follows from. Table it across a few
model sizes and batch sizes so the batch effect on decode is visible.` },

  { slug: 'virtual-function-table', term: 'virtual function table', vol: '880 + 880', kd: '15-16',
    track: 'Deep C++', also: ['virtual table (880/mo) — absorb, do not split'],
    idea: `The vptr is countable in real C++. Compile with the local clang++ and print sizeof for:
an empty struct, one with a single int, the same with one virtual function, with several virtual
functions, with virtual inheritance, and a derived class. The pointer costs 8 bytes once,
regardless of how many virtual functions there are — that surprises people who expect the table
to be per-object. Also print the object layout offsets to show where the vptr sits. Use a real
compiled program via subprocess so the numbers are the compiler's, not a claim.` },

  { slug: 'softmax-function', term: 'softmax function', vol: 5400, kd: 18,
    track: 'LLM internals', also: [],
    idea: `The highest-volume term we do not cover. IMPORTANT: concepts/softmax-vs-sigmoid.md
already exists and must not be duplicated — that page is the two-class comparison; this one is
the function itself. Measure: exact shift invariance (softmax(x) == softmax(x - c) to float64),
the input at which the naive form overflows, and what temperature does to the distribution —
tabulate entropy and max probability against temperature from 0.1 to 10 on a fixed logit vector.
Cross-link the comparison page rather than restating it.` },

  { slug: 'torch-compile', term: 'torch compile', vol: 1300, kd: 17,
    track: 'Compile and export', also: [],
    idea: `Graph breaks are a real integer and torch reports them: torch._dynamo.explain() returns
the break count and the reasons. Write several small functions — one clean, one printing, one
with a data-dependent branch on a tensor value, one calling into numpy — run explain() on each,
and table function against graph-break count and the reported reason. That is the measurement
nobody publishes and it is exactly what decides whether compiling helped.
This page needs torch: put "pip install mlsys-lab torch" in its install line so the checker and
CI install it.` },

  { slug: 'torch-onnx-export', term: 'torch onnx export', vol: 260, kd: 0,
    track: 'Compile and export', also: [],
    idea: `Export a small model to ONNX and measure two things the tutorials skip: the maximum
absolute difference between the torch output and the onnxruntime output on fixed input, and what
happens to a model with a data-dependent control flow — which either fails to export or silently
bakes in the traced branch. Show the baked-in branch by exporting with one input and running with
another that should have taken the other path. That silent wrong answer is the real risk.
This page needs torch and onnxruntime: declare them in its install line.` },
]

const TEMPLATE = `---
title: "What is {TERM}?"
description: "<{TERM} explained, with a measured <metric> you can reproduce, plus a graded exercise.>"
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is {TERM}?

<FIRST 50 WORDS. Sentence 1 defines it with {TERM} as the grammatical subject in the first
four words. Sentence 2 is the consequence, with a number. Sentence 3 forward-references the
measurement.>

## How it works

<300-450 words. Inline-link 4-8 siblings from concepts/ that exist on disk when you check.>

## <heading naming the measurement>

<1-3 sentences on what was varied and what was counted.>

| ... | ... | ... |
|---|---|---|

Reproduce it:

\`\`\`bash
pip install mlsys-lab
python3 - <<'PY'
<the exact lines that produced the table>
PY
\`\`\`

<2-4 sentences reading the table: the pattern, and where it breaks.>

## Practise it

\`\`\`bash
mlsys grade <task-id>
\`\`\`

<Gates quoted from meta.json, what the starter gets wrong, 3-5 more task links.>

## Common mistakes

- **<mistake>.** <why, with the number it costs>

## Where else to practise this

<3-5 entries from RESOURCES.md or LANDSCAPE.md for this track, honest.>

## References

1. <primary source with URL>
2. <primary source with URL>
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['slug', 'gates_pass', 'measurement', 'tasks_linked', 'notes'],
  properties: {
    slug: { type: 'string' },
    gates_pass: { type: 'boolean', description: 'true only if BOTH check_page.py and verify_pages.py print ok' },
    measurement: { type: 'string' },
    tasks_linked: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string', description: 'What you changed about the suggested measurement and why; anything you could not measure; anything the reviewer must re-check.' },
  },
}

const EXISTING = 'memory-coalescing, false-sharing, warp-divergence, cuda-shared-memory-bank-conflicts, '
  + 'cache-blocking, cache-locality, loop-unrolling, simd, kahan-summation, log-sum-exp, '
  + 'bfloat16-vs-float16, integer-quantization-ranges, gguf-vs-safetensors, softmax-vs-sigmoid, '
  + 'rmsnorm-vs-layernorm, gradient-checkpointing, continuous-batching, python-slots, '
  + 'python-descriptors, python-multiprocessing, kmeans, pca'

const CONTEXT = `You are writing one page for github.com/mikamika06/mlsys-lab, a bank of
auto-graded exercises in low-level ML systems. Work in /Users/macbook/mlsys-lab.

=== THE ONE RULE ===
Every number must come from code you ran, and the page must contain the snippet that
regenerates it. Never write a number you have not seen printed. If the snippet prints something
unexpected, the printed value is the truth and the prose is rewritten around it. tools/verify_pages.py
executes your snippet — including plain \`\`\`python fences that import and print — and fails the
page if it prints anything the page does not say.

NO WALL-CLOCK TIMING, ever. Measure counts, exact numeric properties, byte sizes, error
magnitudes. A duration is not reproducible.

=== WHAT THE ENGINE OFFERS ===
  from mlsys.sim import GPU, CudaProgram   -> transactions, smem_waves, divergences, cycles, ...
  from mlsys import cachesim               -> hits, misses, evictions, miss_rate
  from mlsys import scorers                -> max_abs_err, rel_err, mse, size_ratio, ...
Plain deterministic Python, numpy, or a real compiled C++ program run via subprocess are all
equally valid. Several existing pages hand-roll their counter; that is normal here.

If your page needs a package beyond numpy, put it in the page's own \`pip install\` line. The
checker reads that line, and CI installs whatever the pages declare.

=== READ FIRST ===
  concepts/README.md               the house rules
  concepts/memory-coalescing.md    the pattern
  concepts/gguf-vs-safetensors.md  a page whose whole value is actual-vs-nominal
  tools/check_page.py              gate 1 — short, read it
  tools/verify_pages.py            gate 2 — short, read it
  RESOURCES.md                     for your last section

=== TASKS ===
  ls tasks | grep -i <keyword>
  python3 -c "import json;print(json.load(open('tasks/<id>/meta.json'))['gates'])"
Quote gates exactly. Every linked id must exist (test -d tasks/<id>). If no task fits your
concept, say so on the page and link the closest relatives — do not invent an id.

=== BOTH GATES MUST PASS ===
  python3 tools/check_page.py concepts/<slug>.md
  python3 tools/verify_pages.py concepts/<slug>.md
Never edit either tool to pass. Write exactly one file: concepts/<slug>.md.

Existing pages you may link if the file is on disk: ${EXISTING}. Wave-3 siblings are being
written concurrently and may not exist yet.`

phase('Write')

const results = await parallel(CONCEPTS.map((c) => () =>
  agent(
    `${CONTEXT}

=== YOUR CONCEPT ===
slug:   ${c.slug}
term:   ${c.term}     (primary keyword, use verbatim)
track:  ${c.track}
demand: ${c.vol}/mo at KD ${c.kd}
${c.also.length ? 'Also absorb, as headings, do NOT create separate pages:\n' + c.also.map((a) => '  - ' + a).join('\n') : ''}

What to measure — a starting point, not a script. If a better measurement of the same idea
presents itself, take it and say so in notes:
${c.idea}

=== TEMPLATE ===
${TEMPLATE.replace(/\{TERM\}/g, c.term)}

Write concepts/${c.slug}.md, make both gates print ok, and return the JSON.`,
    { label: `page:${c.slug}`, phase: 'Write', schema: SCHEMA, model: 'sonnet' }
  )
))

const ok = results.filter(Boolean)
log(`${ok.filter((r) => r.gates_pass).length}/${CONCEPTS.length} pages passed both gates`)
return {
  passed: ok.filter((r) => r.gates_pass).map((r) => r.slug),
  failed: ok.filter((r) => !r.gates_pass).map((r) => ({ slug: r.slug, notes: r.notes })),
  pages: ok,
}
