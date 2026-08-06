# Incident Report: Triton Dynamic Batching Rejected Requests and Latency Anomalies

## Symptoms
Our production LLM serving endpoints powered by Triton Inference Server are experiencing intermittent request drops and unexpected latency spikes under high load conditions.

Clients have reported receiving HTTP 400 or internal execution errors indicating that requests were dropped or rejected before execution. Concurrently, operational dashboards show significant fluctuations in end-to-end request latencies. The platform team is unable to clearly distinguish whether these latency regressions stem from recent modifications to the Triton dynamic batcher queue configurations (specifically parameters controlling queue sizes and rejection policies) or from general backend model execution slowdowns due to computational bottlenecks.

## Investigation Scope
We need to rigorously evaluate and model queue rejection behaviors when utilizing Triton's `queue_policy` set to `REJECT` alongside strict `max_queue_size` thresholds. Furthermore, we need a diagnostic mechanism to attribute observed latency regressions accurately, separating queue-induced wait time inflation from actual model execution degradation. You are required to implement the core evaluation logic, reference implementations, and automated tests to catch regressions in queue policy handling and latency attribution.
