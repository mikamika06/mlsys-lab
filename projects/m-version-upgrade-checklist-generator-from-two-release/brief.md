Upgrading low-level inference dependencies like vLLM, CUDA driver abstractions, or attention backends frequently introduces silent runtime regressions or breaking changes in flag semantics, model architecture configs, memory footprint defaults, and execution options.

When upgrading between vLLM release note snapshots, engineers often rely on manual skim-reading of release notes and PR logs, which leads to missed deprecations or unhandled breaking behavior during major serving updates.

To prevent silent deployment breaks, we need an automated release-note diff and checklist generator. The generator must take two release-note snapshots (from version A and version B), extract structured API and config change signatures (breaking flags, deprecated parameters, default memory/model shifts, and backend requirements), and produce an actionable upgrade verification checklist.

Your goal is to implement the parsing, signature extraction, diff engine, and checklist generator, along with a regression test suite that catches subtle checklist omission bugs when breaking changes are improperly suppressed.
