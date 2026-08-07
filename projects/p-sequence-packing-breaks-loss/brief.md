# Ticket: Sequence Packing Causes Model Loss Degradation and Context Leakage

Our fine-tuning pipeline recently enabled sequence packing (multipack) to eliminate wasted padding tokens during LLM fine-tuning. Concatenating multiple short training examples into contiguous fixed-length buffers boosted GPU training throughput by more than 2.5x. However, post-tuning model evaluations showed a dramatic drop in downstream instruction-following quality compared to unpacked baselines.

During evaluation, the model frequently generates cross-talk outputs—hallucinating context from unrelated prior examples in the same packed buffer or producing corrupted predictions at sequence transitions. Furthermore, validation loss reported on packed sequences diverges significantly from the ground truth loss computed when evaluating individual sequences independently.

Investigate the packed sequence implementation in `seqpack`. You need to quantify attention leakage across concatenated boundaries, build block-diagonal causal attention masks, correct token loss calculation and normalization across sequence boundaries, verify numerical loss equivalence against unpacked baselines, and ensure high throughput without loss corruption.
