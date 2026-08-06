Ticket: Diagnose Long-Context Failure Modes in NIAH

Our benchmark pipeline is claiming that all our new long-context models are failing the 1M token Needle-In-A-Haystack (NIAH) task due to "RoPE extrapolation" issues. I am highly skeptical of this diagnosis. The models were trained with YaRN scaling, which explicitly prevents out-of-bounds positional embeddings.

Instead, I suspect we are seeing classic attention dilution. As the context length grows to 1 million tokens, the attention distribution becomes too flat. The model still fundamentally knows where the needle is, but the softmax spreads so much probability mass across the massive haystack that the needle's signal washes out completely.

We need a diagnostic tool that definitively proves which failure mode is happening. Please implement two modules.

First, write a tool to measure real attention-entropy growth over a sequence as context scales. If entropy skyrockets, the attention is diluting.

Second, implement a diagnostic module that takes RULER-style accuracy drop-offs and the maximum entropy measurements at those context lengths. It should compute the degradation slope for each model using least-squares, and classify the failure mode at the longest context length. If final accuracy is below threshold and entropy is above threshold, it's `dilution`. If accuracy is low but entropy is also low, it's an actual `rope` failure.
