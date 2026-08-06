We deployed a new NLP model using ONNX Runtime, but the CPU inference latency is completely unacceptable compared to what we saw in initial benchmarking. We strongly suspect there is an issue with our session options or optimization levels.

To investigate, we captured two Chrome Trace profiles using ORT's built-in profiling capabilities. One trace was captured with basic optimizations enabled (`O0`) and the other with extended optimizations (`O99`). The traces are quite large and noisy, so we need a dedicated tool to parse them and surface the structural differences in execution.

You must build a small analyzer tool that ingests these JSON traces and aggregates the operator execution times. The trace format contains many events, but we care specifically about the `Node` category events that are completely finished (phase `X`). These signify actual operator execution.

By parsing both traces and diffing the aggregated execution times and operator counts, we should be able to identify exactly which operators are dominating the basic trace and how much time they consume, and which ones are being successfully optimized away (or fused) in the extended trace.

Parse the duration (`dur`) and the operator name (`args.op_name`), then calculate the absolute differences between the two trace profiles. Sort the final diff list by the duration difference ascending, so we can immediately spot the largest regressions (or improvements). If an operation disappears entirely in the second profile, it must still appear in your diff with negative counts and durations.
