# Ticket: Custom Document and Prefix Masks Cause Out-of-Bounds and Inefficient Block-Sparsity Calculations in FlashAttention

## Symptom
When training or fine-tuning models on packed sequences containing multiple documents or applying Prefix-LM attention patterns using custom mask modifiers (`mask_mod`), several runtime issues and performance inefficiencies have been reported across training clusters.

First, when processing packed documents where multiple distinct sequences are concatenated into a single context window, attention scores incorrectly leak across document boundaries. Tokens from one document are permitted to attend to tokens in preceding or succeeding documents, corrupting downstream generation quality and cross-attention representations.

Second, implementations of Prefix-LM attention modifiers fail to properly establish the causal prefix region where bidirectional attention is allowed among prefix tokens, followed by causal attention for the completion tokens. This results in either complete autoregressive masking (breaking the prefix context) or fully unmasked bidirectional behavior across the entire sequence.

Finally, when attempting to optimize memory and kernel execution via block-sparsity, the current block-sparsity fraction calculator computes incorrect density ratios. It either over-allocates sparse block tables—destroying performance benefits—or under-allocates blocks, leading to CUDA indexing errors and silent data truncation during backward passes. We need a robust, modular implementation for document-masking `mask_mod`, Prefix-LM `mask_mod`, and an accurate block-sparsity fraction calculator to ensure correctness and efficiency.
