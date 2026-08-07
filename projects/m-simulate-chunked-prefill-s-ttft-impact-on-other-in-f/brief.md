Users are complaining that when a large document is pasted into the chat interface, other users who are currently generating text experience a massive and abrupt pause (sometimes up to five seconds) before their next token appears. We are currently using a naive vLLM setup that runs the entire prefill phase for the long document in one go before it resumes any decode steps for other requests.

We need to implement and simulate "chunked prefill", an approach where we cap the number of prompt tokens processed per step. This allows decoding operations for other active requests to interleave with the prefill, keeping the system responsive.

Additionally, we need to analyze our GPU memory utilization from logs to ensure we have enough KV cache blocks to handle this newly interleaved state.

Your tasks are:
1. Parse our simplified vLLM metrics log (CSV format: `timestamp,used_blocks,total_blocks`) to compute mean and maximum memory utilization across the entire log.
2. Build a simulator, `simulate_schedule(prompt_len, inflight_reqs, chunk_size, prefill_cost, decode_cost)`. It must alternate between chunked prefill (processing up to `chunk_size` tokens) and exactly one step of decode for all active in-flight requests. It should return the TTFT (Time To First Token) for the prefill, the maximum stall time experienced by decoding requests, and the total simulated time to clear all workloads.
