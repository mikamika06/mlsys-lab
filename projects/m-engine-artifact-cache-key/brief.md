# Ticket: Production Engine Build, Caching, and Serving Bottlenecks

Our production serving cluster is experiencing severe performance and operational instability during cold starts, dynamic request batching, and continuous integration engine builds. Specifically, the deployment pipeline is frequently rebuilding TensorRT engines from scratch even when model architectures, precision modes, and target GPU profiles are identical, pointing to an unreliable or overly sensitive artifact cache key computation.

Additionally, under high concurrency, our dynamic batching mechanism exhibits erratic tail latencies and poor throughput-versus-delay tradeoffs. The queue-delay threshold parameters are either stalling requests excessively or dispatching under-utilized batches, failing to optimize the balance between queue waiting time and execution latency.

Finally, during service cold starts, initialization profiling shows unpredictable startup spikes. We lack a robust decomposition of the cold-start phase into its constituent steps—ONNX parsing, TensorRT network optimization/tactic selection, weight uploading, and engine serialization—making it impossible to identify where initialization time is consumed or to verify correctness invariants across version upgrades.

We need a clean, modular library in our serving infrastructure to properly compute stable engine artifact cache keys, model dynamic batching queue-delay tradeoffs, and precise cold-start decompositions with comprehensive regression tests to prevent silent regressions in production deployments.
