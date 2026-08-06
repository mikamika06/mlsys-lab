# Ticket: Export Triage Script for PyTorch Models

Production models submitted to our deployment pipeline often fail during `torch.export` attempts. Upstream teams lack a structured report classifying export failures into actionable buckets before escalation to the compiler group.

We need a lightweight triage utility that runs draft export passes on PyTorch GraphModules, captures guard/tracing failure modes, categorizes errors into discrete failure codes, and generates a structured JSON triage report.

## Symptoms
- Export errors produce giant unformatted stack traces that engineers find difficult to parse.
- Uncaptured dynamic shape issues, unsupported Python constructs, and side-effecting operations get mixed together without classification.
- Upstream pipelines cannot programmatically decide whether an issue requires model rewrite vs. graph transform.

## Requirements
1. Implement `triage.exporter.run_draft_export` to run draft export attempts, capturing missing dynamic guard annotations, trace graph mutations, and unsupported dynamic operations.
2. Implement `triage.report.generate_triage_report` to format triage metrics into a structured JSON dictionary containing issue counts, categorized status tags, and remediation priority scores.
3. Write regression tests in `tests/test_regression.py` that verify the triage report correctly catches unclassified failure codes when an unhandled exception type is introduced.
