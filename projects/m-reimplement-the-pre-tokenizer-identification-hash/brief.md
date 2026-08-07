Low-level inference runtimes converting or loading Hugging Face tokenizer definitions into GGUF format often encounter discrepancies caused by pre-tokenizer serialization, hash mismatches, and prompt template alignment errors. Specifically, llama.cpp relies on a deterministic pre-tokenizer identification hash (`tokenizer.ggml.pre` or the underlying string/byte encoding fingerprint) to map Hugging Face tokenizer behaviors accurately. When converting models or constructing prompts directly in low-level runtimes, subtle tokenization divergences—such as divergent byte-fallback handlings or accidental duplication of beginning-of-sequence (BOS) tokens—can severely degrade generation quality or cause silent truncation.

Your task in this exercise unit is to build a verification and analysis pipeline for low-level tokenizer conversion:
1. Reimplement the pre-tokenizer identification hash mechanism used in conversion utilities to fingerprint pre-tokenizer behavior configurations.
2. Measure tokenization disagreement rates between reference Hugging Face tokenizer outputs and target GGUF-mapped tokenizer sequences over a suite of edge-case strings.
3. Detect and diagnose double-BOS token anomalies in rendered chat prompts.

Implement all required modules under `pretokenize/` and ensure your regression suite successfully defends against structural faults.
