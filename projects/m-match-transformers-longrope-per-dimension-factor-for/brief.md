# LongRoPE Scaling Factor & Attention Entropy Diagnostics

Production LLM deployments extending context lengths (e.g., from 4k to 32k/128k) encounter severe perplexity degradation when RoPE positional encodings are naively scaled or misconfigured across dimensions.

We are observing distinct failures across context scaling experiments:
1. In non-uniform RoPE scaling (specifically LongRoPE per-dimension scaling), positional frequencies drop out of sync, causing loss of position discrimination on long contexts.
2. Under 2x to 4x context extensions, static linear position interpolation (Linear-PI) and Dynamic-NTK exhibit different trade-offs against YaRN scaling in token loss / perplexity.
3. Increasing context window without adjusting attention scaling factors leads to attention entropy collapse or spike, degrading the model's focus mechanism.

Your task is to implement the exact LongRoPE per-dimension scaling factor calculation, construct a comparative evaluation of RoPE variants (Linear-PI, Dynamic-NTK, YaRN) on synthetic logit sequences, analyze YaRN's `attention_factor` effect on softmax entropy, and write a robust regression test suite that catches scale factor corruption.
