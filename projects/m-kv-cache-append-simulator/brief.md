We are running into a subtle but catastrophic issue with our new paged KV cache integration during the decoding phase of our inference engine. The symptom is incredibly consistent across all our model architectures: the prefill phase completes successfully, and the very first generated token always makes complete sense. However, starting from the second generated token, the model output instantly diverges into complete gibberish, or it repeats the exact same token endlessly.

Additionally, we have been looking at our hardware utilization metrics. Our memory bandwidth utilization during decode seems unusually low compared to our theoretical estimates. Before we dig into kernel profiling, we need a baseline of what the theoretical memory bandwidth floor actually is for a given batch of sequences.

Your tasks are to:
1. Implement a KV cache append simulator that predicts exactly which block and offset the next token's K and V vectors should be written to, given the current sequence lengths and the block tables. This will help us isolate if we have a routing issue in our memory allocator.
2. Implement a function to calculate the theoretical minimum memory bandwidth required (in bytes) to perform one step of decoding for a batch, accounting for reading the historical KV cache and writing the new token's KV state.
3. Write a regression test that can reliably catch off-by-one errors in the append token logic.
