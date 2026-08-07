# Triton Dynamic Batcher Simulation

Our ML models are seeing latency spikes under load. We use Triton Inference Server's Dynamic Batching, but configuring `preferred_batch_size` and `max_queue_delay_microseconds` is currently guesswork. We need a simulator to visualize how different delays affect throughput and queue latency so we can set these empirically.

Build a discrete-event simulator for Triton's dynamic batcher (assuming a single model instance).

1. **`simulate`**: Process a list of arrival timestamps (in microseconds) and output a list of dispatched batches.
   - The model can process one batch at a time. Time starts at `t=0`.
   - The list of preferred sizes implicitly includes `max_batch_size`.
   - At any time the model is free, evaluate if you can dispatch.
   - You can dispatch if the queue length is `>=` any of the preferred batch sizes. Pick the largest possible preferred size.
   - You can ALSO dispatch if the oldest request in the queue has waited for `>= max_queue_delay_us`. In this case, dispatch as many as possible up to `max_batch_size`.
   - Update the model's busy state using `compute_fn(batch_size)`.

2. **`calculate_metrics`**: Compute throughput (requests per second) and the 99th percentile queue delay (start time minus arrival time) for all requests. The total time for throughput spans from the first arrival to the completion time of the final batch.

3. **`optimize_delay`**: Given a list of delay candidates, return the delay that minimizes the p99 queue delay while keeping throughput `>= throughput_floor`.

Finally, add a regression test safeguarding `optimize_delay` so it never ignores the `throughput_floor`.
