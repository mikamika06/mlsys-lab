An engineering team is running a large-scale inference workload on a cluster with a strict 36GB memory ceiling allocated exclusively for the KV cache per GPU. Recently, during peak traffic loads with long-context prompts, the service suffered a sudden Out-Of-Memory (OOM) crash while attempting to process requests using Multi-Head Attention (MHA).

To prevent these crashes without blindly dropping traffic or downgrading all requests, management wants to evaluate alternative attention architectures—specifically Grouped-Query Attention (GQA) and Multi-Head Latent Attention (MLA)—to determine how much context length each can safely support under the exact same 36GB KV cache budget.

Furthermore, when an OOM traceback occurs in production logs, the team needs a reliable methodology to back-calculate the precise maximum context length that would have safely fit within the 36GB budget, given the exact token count that triggered the failure, the layer count, hidden dimensions, KV heads, and data types.

You need to implement the analytical components that compute KV cache memory consumption for MHA, GQA, and MLA, establish the maximum context limits under the 36GB budget, and back-calculate successful token capacities from OOM logs.
