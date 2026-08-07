# Ticket: Export HF Model to GGUF Quantizations with Automated Quality Verification

## Symptom
Our deployment pipeline requires exporting fine-tuned models from HuggingFace `safetensors` format into GGUF binary models for local execution via llama.cpp.
Currently, our release process stops at training export: we have raw weights in FP16/BF16 precision and no automated mechanism to convert tensors, build tokenizer vocabularies, perform quantized weight conversions, or verify output accuracy.

Engineers are forced to manually invoke disparate scripts without standard metrics on model degradation or throughput.
We lack structured measurements comparing perplexity (PPL), KL-divergence (KLD), and generation speed across quantization recipes such as Q8_0, Q4_0, and importance-matrix (imatrix) guided variants.
Without an end-to-end converter, quantization runner, and evaluation suite, we risk shipping corrupted model binaries, incorrect token mappings, or unverified quantizations that break downstream edge deployments.

## Objective
Implement a robust python library (`gguf_pipeline`) that converts raw HF model tensors and tokenizers into valid GGUF binaries, applies standard and imatrix-guided quantization algorithms, measures perplexity and KL-divergence against the FP16 baseline, benchmarks throughput across recipes, and generates an automated summary report.
