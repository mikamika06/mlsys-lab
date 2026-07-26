export const meta = {
  name: 'absorb-synonyms',
  description: 'Make nine existing concept pages say the phrases people actually type, without splitting into competing pages.',
  phases: [{ title: 'Absorb', detail: 'one agent per page; both gates must still pass' }],
}

// Each page already explains the concept. It just never says the phrase people search for.
// Splitting `logsumexp` off from `log sum exp` would be two pages competing for one query,
// so the synonyms go into the page that already holds the measurement.
const WORK = [
  { page: 'rmsnorm-vs-layernorm.md', gains: 2720,
    phrases: ['layernorm (1,000/mo)', 'rmsnorm (1,000/mo)', 'layer normalization (720/mo)'],
    note: `The comparison phrase this page is titled after is worth 140/mo, while the two single
terms it already explains are 1,000 each. RENAME the page's primary keyword: the title should
lead with the terms that carry the volume while still being a comparison. Something of the shape
"What is layer normalization, and how does RMSNorm differ?" — your call on the exact wording, but
the words "layer normalization", "layernorm" and "rmsnorm" must all appear in the title or an H2,
verbatim.` },

  { page: 'pca.md', gains: 1920,
    phrases: ['pca python (1,600/mo)', 'pca example (320/mo)'],
    note: `This page is titled after the bare word "pca", which has NO measurable search volume,
while "pca python" at 1,600/mo goes unclaimed. That is the single most expensive near-miss in the
set. Retitle it so "PCA in Python" is the primary phrase, and add an H2 that is a worked example,
because "pca example" is a distinct 320/mo query asking for exactly that.` },

  { page: 'kmeans.md', gains: 1170,
    phrases: ['k-means clustering python (390/mo)', 'kmeans sklearn (260/mo)',
              'k-means clustering example (260/mo)', 'k-means clustering examples (260/mo)'],
    note: `"kmeans sklearn" is a real query and the honest answer is a short H2 saying what
sklearn's KMeans does differently from the from-scratch version on this page — it defaults to
k-means++ and n_init>1, which is exactly what this page measures. Do not pretend sklearn is
irrelevant; say what it does and why implementing it once still teaches you something.` },

  { page: 'cache-locality.md', gains: 910,
    phrases: ['temporal locality (260/mo)', 'spatial locality (260/mo)', 'cache line (390/mo)'],
    note: `Temporal and spatial locality are the two halves of this page's subject and deserve one
H2 each, defined precisely. "cache line" is the unit both are measured in and this page already
counts lines — give it a heading too.` },

  { page: 'log-sum-exp.md', gains: 720,
    phrases: ['logsumexp (720/mo)'],
    note: `One spelling, no space. Add it as an H2 or in the first paragraph verbatim. Do NOT
create a second page for it.` },

  { page: 'python-multiprocessing.md', gains: 480,
    phrases: ['python multiprocessing pool (480/mo)'],
    note: `Pool is what people actually reach for, and this page already has the finding that
matters for it: stock Pool never passes buffer_callback, so it always pays the in-band copy for
arrays. Make that an H2 that says "Pool" in its heading.` },

  { page: 'simd.md', gains: 390,
    phrases: ['simd instruction set (390/mo)'],
    note: `An H2 naming the actual instruction sets and their widths — SSE 128-bit, AVX2 256-bit,
AVX-512 512-bit, NEON 128-bit — tied to this page's existing W column, so the abstract width
becomes a real register name.` },

  { page: 'python-descriptors.md', gains: 260,
    phrases: ['descriptor in python (260/mo)'],
    note: `A phrasing variant. One verbatim mention in a heading or the opening is enough.` },

  { page: 'integer-quantization-ranges.md', gains: 260,
    phrases: ['int16 range (260/mo)'],
    note: `This page already tabulates int16. Make sure the exact phrase "int16 range" appears,
and that the row is easy to find.` },
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['page', 'gates_pass', 'phrases_present', 'title_changed', 'notes'],
  properties: {
    page: { type: 'string' },
    gates_pass: { type: 'boolean', description: 'true only if BOTH check_page.py and verify_pages.py print ok afterwards' },
    phrases_present: { type: 'array', items: { type: 'string' }, description: 'The exact phrases now present verbatim in the page.' },
    title_changed: { type: 'string', description: 'The new title line if you changed it, or "unchanged".' },
    notes: { type: 'string', description: 'What you added, what you refused to add and why, anything the reviewer must check.' },
  },
}

const CONTEXT = `You are editing ONE existing page in /Users/macbook/mlsys-lab/concepts/.

=== WHY ===
The page already explains its concept and already carries a measured number. What it does not do
is say the phrase people type into Google. Adding those phrases is worth more traffic than a new
page and costs no new measurement.

=== WHAT YOU MUST NOT DO ===
- Do NOT invent, change or re-round any number. The measurement stays exactly as it is. If you
  think a number is wrong, say so in notes and leave it.
- Do NOT create a second page for a synonym. One page absorbs its variants as headings; two
  pages for one query compete with each other.
- Do NOT keyword-stuff. Each phrase appears where it belongs — a heading, a definition, a
  sentence that a human would write anyway. Every page that currently ranks for these terms
  reads naturally; none is stuffed. A paragraph that exists only to hold a phrase is worse than
  not having the phrase.
- Do NOT touch tools/check_page.py or tools/verify_pages.py. If you edit either, you have failed.
- Do NOT touch any file other than your one page.

=== WHAT MUST STILL BE TRUE AFTERWARDS ===
  python3 tools/check_page.py concepts/<page>      -> ok
  python3 tools/verify_pages.py concepts/<page>    -> ok
The first enforces 600-1400 prose words (so if you add a lot, something has to go — cut padding,
never cut a measurement), front matter, the table, the reproduce block, the exercise link after
the first table, 8-25 resolving relative links, and 2+ references. The second RUNS the page's
snippet and fails if it prints anything the page does not say.

If you change the \`title:\` front matter, change the H1 to match. They must agree.

Read concepts/README.md for the house rules before you start.`

phase('Absorb')

const results = await parallel(WORK.map((w) => () =>
  agent(
    `${CONTEXT}

=== YOUR PAGE ===
concepts/${w.page}      (worth an extra ${w.gains}/mo if these phrases land)

Phrases to absorb, verbatim:
${w.phrases.map((p) => '  - ' + p).join('\n')}

Guidance specific to this page:
${w.note}

Edit the page, make both gates print ok, and return the JSON.`,
    { label: `absorb:${w.page.replace('.md', '')}`, phase: 'Absorb', schema: SCHEMA, model: 'sonnet' }
  )
))

const ok = results.filter(Boolean)
log(`${ok.filter((r) => r.gates_pass).length}/${WORK.length} pages still pass both gates`)
return {
  done: ok.filter((r) => r.gates_pass).map((r) => r.page),
  failed: ok.filter((r) => !r.gates_pass),
  retitled: ok.filter((r) => r.title_changed && r.title_changed !== 'unchanged')
              .map((r) => ({ page: r.page, title: r.title_changed })),
  pages: ok,
}
