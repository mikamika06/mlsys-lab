# KV Cache Memory & Quantization Tradeoffs with GGUF Header Analysis

## Symptom
Your team is deploying LLaMA 3 8B and LLaMA 70B models using `llama.cpp` for long-context inference (`-c 32768`). However, service instances are regularly crashing with Out-Of-Memory (OOM) errors during peak context loads. The deployment team attempts to lower memory usage by enabling Q8_0 KV cache quantization (`--ctk q8_0 --ctv q8_0`), but memory accounting remains inaccurate.

Furthermore, dynamic model loading from GGUF format fails to dynamically compute the required KV cache allocations because the system fails to correctly parse hyperparameter metadata fields from GGUF headers. Finally, stakeholders are concerned that Q8_0 KV quantization might significantly impact perplexity compared to F16, while wondering if Q4_0 KV could offer even higher memory savings without acceptable loss of accuracy.

## Task
1. Implement a GGUF header parser in `kvquant/parser.py` that extracts KV-relevant hyperparameters (`block_count`, `feed_forward_length`, `embedding_length`, `head_count`, `head_count_kv`, and `context_length`) from binary GGUF buffers.
2. Implement KV cache byte calculation and perplexity evaluation logic in `kvquant/memory.py` to compute exact KV cache sizes for context sizes like `-c 32768` under `f16`, `q8_0`, and `q4_0` quantizations, along with predicting perplexity deltas relative to baseline `f16`.
3. Provide regression tests in `tests/test_regression.py` that guard against KV cache under-allocation and erroneous GGUF header parameter calculations.
