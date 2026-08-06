# GGUF Metadata Repair and Decoding Audit

Production text generation pipelines using `llama.cpp` and GGUF model files occasionally run into subtle inference degradation or runtime misconfigurations due to corrupted or inconsistent GGUF KV metadata headers. These issues often manifest without throwing immediate binary parsing errors.

## Symptoms

1. **Premature Truncation or Endless Generation:** After loading a quantized GGUF model, inference runs fail to terminate at proper sentence boundaries or stop immediately after producing a single token. Standard generation checks reveal that the tokenizer vocabulary and model metadata disagree on the Special End-of-Sequence (EOS) token ID.
2. **OutOfMemory or Silently Truncated Contexts:** When attempting to extend context lengths via metadata edits, full file re-quantization or re-conversion is excessively slow. Modifying the GGUF metadata header directly in-place without altering or re-writing massive tensor data buffers is required, but naively overwriting key values corrupts header offsets.
3. **Unexpected RoPE Frequency Scaling Behavior:** Models using extended RoPE scaling (e.g., linear or yarn scaling) produce garbled context beyond their original pre-training window when converted or loaded via different runtimes due to raw metadata key parsing mismatches.

## Objectives

You are tasked with implementing a lightweight GGUF KV metadata parser, manipulator, and validator:
1. Parse raw GGUF metadata dictionaries to detect mismatched EOS token configuration between model architecture keys and tokenizer vocabulary attributes.
2. Implement in-place binary header modification to rewrite `llm.context_length` (or `context_length`) in a target GGUF file buffer/stream without shifting or rewriting tensor data blocks.
3. Decode and normalize standard GGUF RoPE scaling configurations into a unified internal representation.
4. Write a safeguard regression test in `tests/test_regression.py` that verifies EOS validation invariants and catches subtle metadata reconciliation bugs.
