Long prefills are a well-known issue in continuous batching systems. When a new request with a large prompt (e.g., 2000 tokens) arrives, processing it in a single forward pass takes a long time. If we have other requests already in the decode phase, they are forced to wait. This ruins the Inter-Token Latency (ITL) for active users and creates a noticeable "stuttering" effect.

To solve this, modern ML serving engines use **chunked prefill**. We split the long prefill into smaller chunks and pack them together with the decode tokens of other requests up to a fixed `token_budget` per step.

Your task is to implement the `Packer` that maintains this budget and ensures:
1. Decode requests are never starved (they must get priority to maintain a 1-step ITL).
2. Prefill requests are chunked to fill the remaining budget.
3. Prefills are packed in a FIFO manner to avoid unnecessary stretching (don't round-robin the prefill budget across many requests, which would delay all of them).

The total tokens processed in `step()` must never exceed `token_budget`.
