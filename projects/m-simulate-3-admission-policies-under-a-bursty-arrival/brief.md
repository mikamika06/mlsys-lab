The API gateway team is seeing severe timeout cascades during bursty traffic events. When traffic spikes heavily, requests queue up at the inference backend. Because the queue gets so deep, requests are already effectively timed out by the time they are finally popped off the queue and processed by the worker. The worker spends all its computational cycles evaluating requests that the client has already abandoned. We need to implement an admission control layer to proactively drop requests at the door if they are unlikely to be processed in time.

First, we need a reliable simulator that replays a historical trace of arrivals and processing costs. You will implement three admission policies to see their effects on our traffic:
1. `accept_all`: Unbounded queue.
2. `queue_limit`: Drop the request immediately if the number of currently active, pending requests in the queue equals or exceeds `max_len`.
3. `time_limit`: Drop the request immediately if its expected wait time (before execution begins) strictly exceeds `max_wait`.

Second, to properly tune `max_wait`, we need to analyze historical outage logs to find the exact request that triggered the timeout cascade. Given an unbounded trace and a target `max_wait`, write a function to identify the first request ID whose expected wait time crossed the threshold.
