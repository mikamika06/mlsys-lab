export const meta = {
  name: 'fix-contradicting-examples',
  description: 'Thirty-three task statements whose worked example disagrees with their own reference. Decide which side is wrong from the statement itself, fix that side.',
  phases: [{ title: 'Fix', detail: 'one agent per task; check_examples.py must then agree' }],
}

const TASKS = [
  "cpu-packed-vs-aligned-footprint-for-n-records",
  "gpu-compute-arithmetic-intensity-of-an-op",
  "llm-classify-pre-tokenization-regex-splits",
  "llm-gqa-effect-on-decode-arithmetic-intensity",
  "llm-no-repeat-ngram-blocking",
  "llm-real-gpu-hbm-utilization-for-decode-kernel",
  "num-ai-of-mxkxn-matmul-with-without-reuse",
  "num-predict-broadcast-result-shape",
  "num-predict-ieee-special-value-expressions",
  "num-report-machine-epsilon-for-fp32",
  "num-spot-the-cancelling-subtraction",
  "rwa-classify-each-past-token-as-sink-window-evicted",
  "rwb-kv-cache-transfer-bytes-for-a-request",
  "rwb-would-this-pair-share-a-prefix-node",
  "rwm-memory-of-int4-kv-fp16-residual-window-vs-full-fp16-cache",
  "rwm-optimizer-memory-adam-fp32-vs-8-bit-blockwise-vs-paged",
  "rwm-peak-attention-memory-naive-o-n-2-vs-flash-o-n",
  "rwm-peak-gpu-bytes-total-h2d-transfer-for-a-k-layer-window",
  "rwm-peak-vram-full-resident-vs-model-vs-sequential-offload",
  "rwm-per-gpu-bytes-for-params-grads-optimizer-given-phi-n-stage",
  "rwq-mixed-precision-memory-accounting"
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['task', 'which_was_wrong', 'what_changed', 'checker_passes', 'verify_passes', 'notes'],
  properties: {
    task: { type: 'string' },
    which_was_wrong: {
      type: 'string',
      enum: ['statement', 'reference', 'both', 'neither-checker-was-wrong'],
      description: 'Which side you concluded was wrong, judged from the statement\'s own stated rule.',
    },
    what_changed: { type: 'string', description: 'Exactly what you edited, old value -> new value.' },
    checker_passes: { type: 'boolean', description: 'python3 tools/check_examples.py <task> reports no disagreement' },
    verify_passes: { type: 'boolean', description: 'bash tools/verify_task.sh <task> prints TASK_OK' },
    notes: { type: 'string', description: 'Your reasoning, and anything the reviewer must re-check.' },
  },
}

const CONTEXT = `You are fixing ONE task in /Users/macbook/mlsys-lab/tasks/.

=== THE DEFECT ===
The worked example in \`task.md\` disagrees with what \`solution_ref.py\` actually returns. A
learner copies that example first, so a wrong one makes them implement the wrong thing and fail
a task they had understood correctly.

See it:
    python3 tools/check_examples.py <task-id>

=== YOUR JOB IS TO DECIDE WHICH SIDE IS WRONG ===
Do NOT assume the reference is right. Work it out from the statement's own stated rule — the
formula, the definition, the explanation under the example. Three real cases from this batch:

  - \`rwm-per-gpu-bytes...\`: example said 12500000, reference returned 125000000, and the
    statement's OWN explanation derives "125 M bytes". The example had a digit dropped. Fix the
    example.
  - \`rwm-peak-attention-memory...\`: example said (256, 192) and the explanation wrote
    "3 x 2 x 4 x 3 x 8 = 192" — but that expression is 576, which is what the reference returns.
    The statement was wrong twice, in the example and in its own arithmetic. Fix both.
  - Sometimes the reference is the wrong one. If the statement's rule is clear and the reference
    contradicts it, fix \`solution_ref.py\` — and then \`starter.py\` must still fail, so check.

Whichever side you change, the statement, its explanation and the reference must all agree
afterwards. Do not paper over a disagreement by deleting the example.

=== AFTER YOUR FIX, BOTH MUST HOLD ===
    python3 tools/check_examples.py <task-id>     # no disagreement reported
    bash tools/verify_task.sh <task-id>           # prints TASK_OK
The second matters: it re-checks that the reference passes the gates AND the shipped starter
still fails them. If you changed the reference and the starter now passes, the task no longer
discriminates and you have broken it.

=== SCOPE ===
Touch only files inside tasks/<task-id>/. Never edit tools/check_examples.py or any checker —
if the checker is genuinely wrong about this task, do not edit it, report that in notes with
which_was_wrong = "neither-checker-was-wrong" and leave the task alone.`

phase('Fix')

const results = await parallel(TASKS.map((t) => () =>
  agent(
    `${CONTEXT}

=== YOUR TASK ===
${t}

Read tasks/${t}/task.md and tasks/${t}/solution_ref.py, run the checker to see the
disagreement, decide which side is wrong, fix it, and confirm both commands pass.
Return the JSON.`,
    { label: `fix:${t.slice(0, 34)}`, phase: 'Fix', schema: SCHEMA, model: 'sonnet' }
  )
))

const ok = results.filter(Boolean)
const fixed = ok.filter((r) => r.checker_passes && r.verify_passes)
log(`${fixed.length}/${TASKS.length} now agree with their reference and still verify`)
return {
  fixed: fixed.map((r) => ({ task: r.task, wrong: r.which_was_wrong, change: r.what_changed })),
  unresolved: ok.filter((r) => !(r.checker_passes && r.verify_passes))
                .map((r) => ({ task: r.task, wrong: r.which_was_wrong, notes: r.notes })),
  by_side: ok.reduce((m, r) => ({ ...m, [r.which_was_wrong]: (m[r.which_was_wrong] || 0) + 1 }), {}),
}
