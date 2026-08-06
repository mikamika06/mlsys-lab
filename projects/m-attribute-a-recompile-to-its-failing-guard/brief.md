# Recompilation Profiling and Guard Attribution

Production workloads using PyTorch TorchDynamo and JIT compilers frequently suffer from silent performance degradation due to hidden recompilations. A dynamic batching pipeline has started experiencing high latency spikes during live traffic. Initial telemetry shows that the graph compiler is repeatedly throwing away compiled artifacts and falling back to tracing, but standard logs only report that recompilation occurred—they fail to identify which guard failed, what input property triggered the invalidation, or how many total recompiles accumulated across the dynamic batching sequence.

Your task is to implement an attribution mechanism and guard evaluator for traced graphs. You will inspect dynamic guard conditions (such as batch dimensions, dynamic sequence lengths, and dtypes) against input streams to trace exactly which guard evaluation failed and caused a graph specialization failure. Finally, you will calculate the cumulative recompile count across a scheduled stream of dynamic batch sizes and write regression tests to detect unhandled guard failures.

## Milestones

1. Implement `attribute_guard_failure(guards, inputs)` in `guard_profiler/attribution.py` to evaluate dynamic guard expressions against input schemas and identify the precise path and expression that failed.
2. Implement `simulate_batch_schedule(graph_spec, batch_sizes)` in `guard_profiler/evaluator.py` to calculate the total recompile count and trace history given a sequence of incoming dynamic batch dimensions.
3. Write regression tests in `tests/test_regression.py` that validate guard failure attribution and verify that unexpected shape changes or unhandled guard conditions are correctly caught.
