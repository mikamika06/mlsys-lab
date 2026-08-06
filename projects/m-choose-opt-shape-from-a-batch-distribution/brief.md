# Ticket: Latency Spikes and Random Enqueue Failures Under Production Workloads

We are seeing two separate issues in our TensorRT runtime deployment for batch inference on dynamic shapes. 

First, under fluctuating production traffic, our execution latency drops significantly when batch sizes fluctuate around our current optimization profile settings. We need a way to pick optimal shapes (`opt_shape`) for execution profiles based on empirically observed batch distributions (e.g., p50, p90, or mode targets depending on workload priority) rather than relying on arbitrary manual guesses.

Second, under high-throughput spikes, the inference engine crashes with an out-of-profile enqueue error (`[Error] Enqueue failed: input shape outside profile boundaries`). This happens because incoming batches occasionally fall outside the `[min_shape, max_shape]` range configured for the active profile plan.

We need a clean, structured package `trtprof` that:
1. Computes the optimal profile shapes (`min`, `opt`, `max`) given a distribution of observed batch sizes and a user strategy.
2. Generates an executable profile plan mapping workload specifications to valid TRT dynamic shape bounds.
3. Implements a runtime fallback and profile switcher mechanism that catches out-of-profile shapes, selects or adjusts to an admissible profile without raising runtime enqueue errors, and provides regression tests to ensure out-of-profile boundaries are caught safely.
