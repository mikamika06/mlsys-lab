Our production computer vision models are experiencing unacceptable tail latency spikes. An analysis of the trace logs reveals that the GPUs are frequently underutilized, running with batch sizes of 1 or 2 because requests aren't perfectly aligned in time. This wastes massive amounts of compute.

We enabled Triton Inference Server's dynamic batching to group incoming requests, but tuning its configuration blindly has been disastrous. If we set `max_queue_delay_microseconds` too high, the P99 latency spikes because early requests wait too long. If we set it too low, batching never triggers and throughput collapses.

We need a discrete-event simulator that models Triton's dispatch algorithm so we can optimize these parameters against real production traces.

You must build:
1. **The Simulator**: Recreate Triton's dispatch logic.
   - Time starts at 0. You receive sorted request arrival times (in microseconds).
   - If the model is free and `len(queue) >= p` for any `p` in `preferred_batch_sizes` (or `p == max_batch_size`), dispatch immediately using the largest `p` that fits.
   - If the oldest request in the queue has waited `>= max_queue_delay_us`, dispatch up to `max_batch_size` immediately.
   - Otherwise, requests wait.
2. **The Metrics Engine**: Calculate `throughput_req_sec` (total requests / total time from first arrival to last completion) and the `p99_queue_delay_us`.
3. **The Optimizer**: Sweep a list of preferred sizes and delay candidates to find the configuration that minimizes P99 queue delay, subject to a strict `throughput_floor`.
