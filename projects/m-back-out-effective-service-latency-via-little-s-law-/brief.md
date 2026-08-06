Production metrics for our LLM serving engine `llm-capacity-planner` show significant discrepancies between predicted and actual token generation costs under tight SLA constraints.

Current auto-scaling mechanisms trigger premature scale-outs, leading to over-provisioning and inflated cloud compute spend ($/1K tokens). Simultaneously, naive capacity models based on static peak-latency estimates fail to capture true system concurrency, causing SLA violations when traffic bursts occur.

Engineers report that batching configurations are chosen arbitrarily, without accounting for how batch size alters effective service latency and server utilization. We need a mathematically grounded capacity planning toolkit to extract operational constants from runtime traces, find optimal batching strategies under SLA limits, and evaluate autoscaler efficiency.

You are tasked with implementing the capacity planning module in `capacity/`:
1. `capacity/trace.py`: Deduce effective service latency $W$ from continuous concurrency $L$ and arrival rate $\lambda$ recorded in production logs using Little's Law ($L = \lambda W$).
2. `capacity/batching.py`: Sweep batch sizes to identify the exact batch size that minimizes cost per 1000 tokens ($/1K tokens) while respecting a hard latency SLA.
3. `capacity/autoscaler.py`: Calculate the theoretical minimum replica count required for a given target load and benchmark it against real autoscaling trace logs to quantify over-provisioning slack.
4. `tests/test_regression.py`: Implement a regression test suite that validates these capacity calculations and detects bugs in SLA constraint enforcement.
