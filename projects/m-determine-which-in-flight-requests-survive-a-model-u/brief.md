# In-Flight Request Survival and Drain Timeout Calculator

During blue-green model updates and version rollouts on Triton Inference Server instances, clients frequently report intermittent `503 Service Unavailable` errors, truncated inference streams, and unexpected TCP connection resets. Downstream API gateways register thousands of canceled requests whenever an unload command is dispatched to model instances under active load.

Engineers attempted to resolve this by configuring static unload timeouts across all serving pods. However, during high-throughput inference surges, requests with large batch sizes or long sequence completion steps are still killed prematurely before completing execution. Conversely, setting large conservative timeouts unnecessarily holds GPU memory during rolling deployments, stalling cluster-wide model updates and violating deployment SLAs.

You must implement a dynamic drain analyzer for Triton model control operations. First, calculate which in-flight requests will survive model unload signals depending on execution progress, model unload mode, and grace periods. Second, derive the exact minimum drain timeout required to allow all in-flight requests to complete safely without losing data. Finally, construct a regression test suite that verifies whether drain timeout logic properly accounts for dynamic queue latency under queued batch states.
