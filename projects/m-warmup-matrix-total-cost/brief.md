Lately, our production serving instances have been consistently missing their SLA (target p99 latency is 250ms), leading to upstream request timeouts. We strongly suspect our dynamic batching parameters are misconfigured. Right now, we are just guessing the maximum batch size and timeout values. When load spikes, requests either wait in the queue for too long, or we attempt to process an oversized batch which takes too long to compute.

Furthermore, whenever we reboot an instance, the server hangs for minutes doing a CUDA graph warmup. We need to accurately estimate the total time it will take to warm up a given Cartesian product of batch sizes and sequence lengths so we can tune our Kubernetes readiness probes.

We need a module that:
1. Models the execution time (latency) for a batch size and sequence length. The empirical model is: `10.0 + 0.5 * batch_size + 0.1 * seq_len + 0.05 * batch_size * seq_len`.
2. Computes the total execution time needed to run exactly one forward pass for every combination of provided batch sizes and sequence lengths.
3. Simulates a request queue to predict p50 and p99 latencies for given parameters.
4. Tunes the maximum batch size and wait timeout for an arrival stream to meet a strict p99 target.
