# How a Part-2 unit is built

A unit is one directory under `projects/<id>/`. M and L differ only in size:
M is 2-3 milestones and 3-6 files the learner edits, L is 7 milestones and 8-30.

```
projects/<id>/
  project.json     metadata, milestones, gates
  brief.md         the ticket: a symptom, not a diagnosis
  skeleton/        what the learner gets; every function raises NotImplementedError
  reference/       our working solution, same layout as skeleton/
  harness/mN.py    one checker per milestone: check(workdir) -> dict of metrics
```

Everything is written in **English**. The README, the 2053 task statements and the
Marketplace listing are English; a translated unit is an inconsistency, not a
localisation.

## The contract, which a machine checks

`python3 tools/verify_project.py <id>` must print
`reference N/N, skeleton 0/N`. Both halves are required:

- **the reference clears every milestone** — otherwise the work is impossible;
- **the skeleton clears none** — otherwise the gate measures nothing.

## Rules

**A gate is never wall-clock time.** Allowed: an invariant (something does not
raise, a counter is zero, output matches within a tolerance), a ratio against the
learner's own baseline, or the analysis of a recorded artifact. Forbidden:
`time.time()` in a gate, or a threshold tuned to one machine.

**The grader computes its own reference.** No hard-coded expected answers: either
an oracle (a real library, `mlsys.sim.*`) or a recomputation from the same inputs.

**The brief states a symptom, not the defect.** "p99 went up after we enabled X",
not "remove the graph break in forward()".

**No comments in code.** Explanation belongs in `brief.md`.

**Determinism.** Fixed seed, integer time, no network. Generate fixtures in
`harness/`; never commit binaries.

**Tier.** T0 is pure python/numpy and runs anywhere. T1 needs a real library:
list it in `project.json` as `requires_pkgs`, and have the checker return a `_note`
saying so rather than crashing. The verifier skips a unit whose declared package is
absent, so an honest declaration is what keeps CI green.

**The last milestone is always a guardrail**: the learner writes a test, the checker
replaces something in their own code with a broken version, and the test has to
fail. The injected fault must break an **invariant** — not merely be a different
valid implementation. An injection that is arguably better than the original is not
a fault, and the milestone will pass for the wrong reason or never pass at all.

## Checker shape

```python
def check(workdir):
    return {"metric_name": 1.0, "_note": "optional explanation shown to the learner"}
```

Gates in `project.json`:

```json
{"metric": "metric_name", "op": "==", "threshold": 1}
```

`op` is one of `== <= >= < >`. A metric the checker did not return counts as failed.

## What the spec gives you

`tools/specs2/<id>.json` carries `area`, `track`, `tier`, `gate_metric` and the
`ideas` the unit is assembled from. Cover those ideas with the milestones rather
than inventing a different topic.
