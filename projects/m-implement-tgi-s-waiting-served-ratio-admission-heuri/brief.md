We are deploying a new continuous batching router for our large language models, drawing heavily from the Text Generation Inference (TGI) architecture. Under moderate traffic, we are seeing sporadic Out of Memory (OOM) crashes on our inference workers, even though we strictly configured max batch token constraints.

The continuous batching router manages both incoming queries (the prefill phase) and ongoing generations (the decode phase). TGI utilizes a waiting-served ratio admission heuristic to prevent queue starvation while protecting the active batch limit. The heuristic blocks admission if the queue is small relative to the active batch, letting the model focus purely on fast decoding.

You need to implement three things to help us resolve the OOMs and observe our router behavior:

1. Implement the waiting-served ratio admission heuristic. We provide a queue of waiting requests and the active batch state, and you must admit requests up to the max prefill and max total limits.
2. Build a utility function that parses a recorded router log and computes the average batch-token utilization across all ticks. We define utilization at a given tick as the sum of all token lengths (active inputs, active generated, and newly admitted prefills) divided by the absolute maximum token capacity.
3. We suspect the OOM is due to an accounting flaw in the total tokens boundary check. Write a targeted regression test that validates the admission control explicitly tracks the tokens generated over time, keeping the batch strictly bounded.
