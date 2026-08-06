Our CPU inference service pods are restarting constantly during deployments, and user experience is suffering due to what looks like a complete misconfiguration of the inference engine.

There are two distinct symptoms. First, whenever a pod starts, the very first request takes upwards of 5 seconds to process, causing readiness probes to time out. Our graph compilation isn't being cached across restarts because the cache directory functionality isn't implemented.

Second, our background job workers send large batches (up to 64 items), but the server processes them sequentially, heavily underutilizing the 32-core nodes. We requested a `throughput` mode, but it behaves identically to `latency` mode, which only uses a single execution stream.

We need to formalize the engine configuration so that `latency` mode forces 1 stream (allocating all cores to threads), while `throughput` mode provisions 1 stream per 4 cores (minimum 1 stream). Then, wire up the cache directory mock to persist compiled binaries. Finally, build an estimator to prove that throughput mode mathematically outperforms latency mode on large batches.
