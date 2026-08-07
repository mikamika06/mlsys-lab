# Investigating Un-evaluated Graph Retention Memory Leaks and Compilation Churn

Our edge inference microservice on Apple Silicon nodes is failing under long-running production traffic due to progressive process Resident Set Size (RSS) memory growth and unpredictable execution latency spikes.

During continuous serving, the RSS footprint grows unbounded across successive inference iterations even though batch sizes remain stationary. Operational logs indicate that session contexts and request histories store intermediate array handles without triggering evaluation or explicit graph clearing, inadvertently pinning un-evaluated lazy computation graphs in memory.

At the same time, when processing variable sequence lengths or dynamic input shapes, the service experiences significant latency penalties. We need to characterize the exact compilation and recompilation overheads incurred by compiled lazy evaluation graphs under shifting input shapes, comparing performance profiles against PyTorch MPS (`torch.compile(backend='aot_eager')`).

To isolate and prevent these issues, you will implement a deterministic profiling framework to:
1. Simulate lazy graph evaluation and reproduce RSS growth caused by holding references to un-evaluated computation graphs versus evaluating and releasing them.
2. Build comparative graph compilation profiles benchmarking execution overhead between MLX compiled graphs (`mx.compile`) and PyTorch MPS (`torch.compile(backend='aot_eager')`).
3. Quantify recompilation costs incurred when input shapes vary, and implement regression tests to catch un-cleared graph handles.
