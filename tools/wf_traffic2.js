export const meta = {
  name: 'traffic-pass-2',
  description: 'Absorb another ~1,700/mo of synonyms into existing pages, and add five pages for terms with no home.',
  phases: [{ title: 'Cover', detail: 'absorb into nine pages, write five new ones' }],
}

// Phrases people type that an existing page already explains but never says.
const ABSORB = [
  { page: 'softmax-function.md', gains: 490,
    phrases: ['softmax temperature (210/mo)', 'softmax function python (70/mo)',
              'softmax function formula (70/mo)', 'softmax vs relu (50/mo)',
              'masked multi head attention (50/mo)', 'multi head attention pytorch (140/mo)'],
    note: `This page already measures temperature against entropy — the phrase "softmax
temperature" should be a heading over that. "softmax function formula" wants the formula on
screen: make sure it is displayed, not just described. Only claim the PyTorch and masked-attention
phrases if the page genuinely says something true about them; if it does not, skip those two and
say so in notes rather than bolting on a paragraph.` },

  { page: 'bfloat16-vs-float16.md', gains: 140,
    phrases: ['fp16 vs fp32 (140/mo)'],
    note: `The page already tabulates float32 alongside the two 16-bit formats. One heading and a
sentence comparing fp16 to fp32 directly is enough.` },

  { page: 'memory-coalescing.md', gains: 40,
    phrases: ['coalesced memory access (40/mo)'],
    note: `A phrasing variant of the page's own title. One verbatim use in a heading or the
opening paragraph.` },

  { page: 'cuda-shared-memory-bank-conflicts.md', gains: 190,
    phrases: ['cuda shared memory (140/mo)', 'constant memory cuda (50/mo)'],
    note: `"cuda shared memory" is the parent concept this page is a detail of — it deserves a
heading that defines shared memory before the conflict mechanics. Add constant memory only if you
can say something true and short about how it differs; otherwise skip it and say so.` },

  { page: 'integer-quantization-ranges.md', gains: 320,
    phrases: ['int4 range (40/mo)', 'int8 quantization (90/mo)',
              'post training quantization (140/mo)', 'pytorch quantization (140/mo)'],
    note: `The first two are already tabulated; make the phrases appear. "post training
quantization" is the family this page's arithmetic belongs to — one heading placing it. Claim
"pytorch quantization" only if you add something true about the PyTorch API surface; otherwise
skip and say so.` },

  { page: 'false-sharing.md', gains: 40,
    phrases: ['false sharing cache (40/mo)'],
    note: `A variant of the title. One natural use.` },

  { page: 'simd.md', gains: 160,
    phrases: ['simd vectorization (70/mo)', 'what is simd in computer architecture (40/mo)',
              'simd optimization (40/mo)'],
    note: `Three phrasings of what this page already is. Headings, not a keyword list.` },

  { page: 'rmsnorm-vs-layernorm.md', gains: 340,
    phrases: ['layernorm vs batchnorm (140/mo)', 'layer normalization vs batch normalization (110/mo)',
              'layernorm pytorch (140/mo)'],
    note: `Batch normalization is the comparison people actually reach for first, and this page
never mentions it. A heading contrasting the axis each normalises over — batch vs feature — is
genuinely useful and is what those two queries want. Say what torch.nn.LayerNorm does for the
PyTorch phrase.` },

  { page: 'kmeans.md', gains: 220,
    phrases: ['kmeans python (170/mo)', 'pca python example (50/mo)'],
    note: `"kmeans python" is a near-duplicate of a phrase already here; make it verbatim. Skip
the PCA phrase — it belongs on pca.md, not here — and say so in notes.` },
]

// Terms with real demand, low difficulty, and no existing page to absorb them.
const NEW = [
  { slug: 'cuda-graphs', term: 'cuda graphs', vol: 300, kd: 0, track: 'GPU / CUDA',
    also: ['pytorch cuda graph (40/mo)', 'torch compile dynamic (40/mo)'],
    idea: `A CUDA graph replaces per-launch CPU work with one replay. The measurable thing here is
launch COUNT, not time: for a sequence of N small kernels, count the host-side launches issued
without graphs (N per iteration) against with graphs (1 replay per iteration), across iteration
counts. Then the constraint that decides whether you can use them at all — the captured shapes
and pointers are frozen — which is why dynamic shapes force re-capture. Count re-captures for a
workload whose shape changes every k steps.` },

  { slug: 'move-semantics', term: 'move semantics', vol: 310, kd: '0-6', track: 'Deep C++',
    also: ['rvalue reference (170/mo) — absorb, do not split', 'smart pointers vs raw pointers (40/mo)'],
    idea: `Count copies and moves. Write a real C++ program with a type that instruments its copy
constructor, move constructor, copy assignment and move assignment, run it through the local
clang++, and table how many of each a given operation performs: pass by value, pass by const
reference, return a local, push_back an lvalue, push_back a std::move'd lvalue, and a vector
growth that reallocates. The number people get wrong is the last one — whether reallocation moves
or copies depends on the move constructor being noexcept. Show both.` },

  { slug: 'kernel-fusion', term: 'kernel fusion', vol: 140, kd: 0, track: 'Compile and export',
    also: [],
    idea: `Fusion exists to stop round-tripping through global memory. Measure it with the software
GPU: an unfused chain of elementwise kernels each reads and writes global memory, a fused one reads
once and writes once. Table global-memory transactions and mem_insts against chain length for both.
The ratio approaches the chain length, which is the whole argument for fusion, and the transactions
counter shows it exactly.` },

  { slug: 'self-attention-vs-cross-attention', term: 'self-attention vs cross attention', vol: 210, kd: 4,
    track: 'LLM internals', also: [],
    idea: `The difference is which sequence supplies K and V, and it shows up in shapes and in
cache behaviour. Table, for fixed dimensions: the shapes of Q, K, V and the score matrix for
self-attention over length L against cross-attention from a decoder of length L into an encoder of
length S; the parameter count of each (identical); and the KV-cache bytes each needs during
incremental decoding — which is the real difference, because cross-attention's K and V are computed
once from the encoder and never grow, while self-attention's grow every step.` },

  { slug: 'llm-inference-optimization', term: 'llm inference optimization', vol: 110, kd: 0,
    track: 'LLM systems', also: ['kv cache compression (110/mo)'],
    idea: `A map page, but it must still carry a number. For one fixed model shape, table what each
lever actually buys, computed not asserted: KV-cache bytes before and after switching MHA to GQA;
before and after quantising the cache to int8; the decode arithmetic intensity before and after
batching; and the memory a paged allocator saves against contiguous pre-allocation. Each row links
the page that measures it properly. The point is the RELATIVE size of the levers, which nobody
tabulates side by side.` },
]

const SCHEMA_A = {
  type: 'object', additionalProperties: false,
  required: ['page', 'gates_pass', 'phrases_added', 'phrases_refused', 'notes'],
  properties: {
    page: { type: 'string' },
    gates_pass: { type: 'boolean' },
    phrases_added: { type: 'array', items: { type: 'string' } },
    phrases_refused: { type: 'array', items: { type: 'string' },
                       description: 'Phrases you deliberately did NOT add because the page had nothing true to say about them.' },
    notes: { type: 'string' },
  },
}

const SCHEMA_N = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'gates_pass', 'measurement', 'tasks_linked', 'notes'],
  properties: {
    slug: { type: 'string' },
    gates_pass: { type: 'boolean' },
    measurement: { type: 'string' },
    tasks_linked: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const RULES = `Work in /Users/macbook/mlsys-lab.

Read first: concepts/README.md (house rules), tools/check_page.py and tools/verify_pages.py
(the two gates — both short, read them), RESOURCES.md (for the "where else" section).

BOTH GATES MUST PASS on your file when you are done:
    python3 tools/check_page.py concepts/<file>
    python3 tools/verify_pages.py concepts/<file>
Never edit either tool to make your page pass. Touch exactly one file.

NO WALL-CLOCK TIMING anywhere, ever. Counts, exact numeric properties, byte sizes, error
magnitudes. A duration is not reproducible and nothing here is graded on one.

Never write a number you have not seen printed. verify_pages.py runs your snippet — including
plain \`\`\`python fences that import and print — and fails the page if it prints anything the
page does not say. If your page needs a package beyond numpy, put it in the page's own
\`pip install\` line; the checker reads that line.`

phase('Cover')

const absorbed = parallel(ABSORB.map((w) => () =>
  agent(
    `${RULES}

=== ABSORB SYNONYMS INTO AN EXISTING PAGE ===
concepts/${w.page}   (worth ~${w.gains}/mo if these land)

The page already explains this material. It just never says the phrases people type. Adding
them is worth more than a new page and needs no new measurement.

Phrases:
${w.phrases.map((p) => '  - ' + p).join('\n')}

Guidance:
${w.note}

HARD RULES
- Do not invent, change or re-round any number. The measurement stays exactly as it is.
- Do not create a second page for a synonym; one page absorbs its variants as headings.
- Do not keyword-stuff. Each phrase goes where a human writer would naturally put it. A
  paragraph that exists only to hold a phrase is worse than not having the phrase — if the page
  has nothing true to say about a phrase, REFUSE it and list it in phrases_refused.
- The word-count ceiling is 1400 prose words. If you need room, cut padding, never a measurement.
- If you change \`title:\`, change the H1 to match.

Return the JSON.`,
    { label: `absorb:${w.page.replace('.md', '')}`, phase: 'Cover', schema: SCHEMA_A, model: 'sonnet' }
  )
))

const written = parallel(NEW.map((c) => () =>
  agent(
    `${RULES}

=== WRITE A NEW PAGE ===
slug:   ${c.slug}
term:   ${c.term}     (primary keyword, verbatim)
track:  ${c.track}
demand: ${c.vol}/mo at KD ${c.kd}
${c.also.length ? 'Absorb as headings, do NOT create separate pages:\n' + c.also.map((a) => '  - ' + a).join('\n') : ''}

What to measure — a starting point. If a better measurement of the same idea presents itself,
take it and say so in notes:
${c.idea}

Follow the shape of concepts/memory-coalescing.md exactly: front matter with
title/description/datePublished/dateModified/author, H1 matching the title, a 50-word opening
whose first sentence is a standalone definition, "## How it works", a heading naming the
measurement with a markdown table under it, a "Reproduce it:" bash block containing
\`python3 - <<'PY' ... PY\`, 2-4 sentences reading the table, "## Practise it" with a
\`mlsys grade <task-id>\` line AFTER the first table, "## Common mistakes", "## Where else to
practise this", and "## References" with 2+ numbered items carrying URLs. 8-25 relative links,
all of which must resolve — check task ids with \`test -d tasks/<id>\` before linking.

If no task in the bank fits your concept, say that on the page and link the closest relatives
rather than inventing an id.

Return the JSON.`,
    { label: `page:${c.slug}`, phase: 'Cover', schema: SCHEMA_N, model: 'sonnet' }
  )
))

const [A, N] = await Promise.all([absorbed, written])
const a = A.filter(Boolean), n = N.filter(Boolean)
log(`absorbed ${a.filter((r) => r.gates_pass).length}/${ABSORB.length}, wrote ${n.filter((r) => r.gates_pass).length}/${NEW.length}`)
return {
  absorbed: a.map((r) => ({ page: r.page, ok: r.gates_pass, added: r.phrases_added, refused: r.phrases_refused })),
  written: n.map((r) => ({ slug: r.slug, ok: r.gates_pass, tasks: r.tasks_linked })),
  failures: [...a, ...n].filter((r) => !r.gates_pass),
}
