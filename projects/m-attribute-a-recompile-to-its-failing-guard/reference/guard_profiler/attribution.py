def attribute_guard_failure(guards, inputs):
    """Evaluates guard conditions against tensor inputs to find the failing guard."""FILE: brief.md
```markdown
# Attribute a recompile to its failing guard

We are investigating unexplained recompilation cascades in an inferencing and fine-tuning pipeline. Batches of varying shapes and tensor metadata are passing through compiled execution graphs, but our latency telemetry spikes unpredictably because Dynamo/Inductor traces new graphs when guard checks fail. Currently, we can only see that recompilations occur, but we cannot identify which specific guard condition triggered the recompile or predict how many recompilations will occur under a given batch schedule.

Your goal is to build an attribution engine and guard evaluator for execution graphs.

First, implement a guard evaluation mechanism in `guardeval/evaluator.py`. The evaluator must maintain graph guard definitions (such as dynamic batch dimensions, dtype matching, and tensor strides/contiguity), evaluate incoming tensor metadata against registered guard trees, identify the first failing guard condition, and record guard failure telemetry.

Second, in `guardeval/attribution.py`, simulate an execution stream given a batch sequence schedule. You must track active compiled graphs, determine whether an existing graph satisfies all guards for an incoming input, attribute recompilations directly to the specific guard failure when no candidate graph passes, and return the total recompile count along with detailed guard failure logs.

Third, write regression tests in `tests/test_regression.py` that validate guard attribution accuracy. Your test suite must pass on a correct implementation and fail if an engine incorrectly ignores stride mismatch guards during guard evaluation.
