# GGUF LoRA Adapter Conversion and Weight Dequantization

## Background
The high-throughput LLM inference pipeline in `gguf_adapter/` is failing during fine-tuned model evaluation. Downstream benchmark runners report severe output degeneration (high perplexity or garbage tokens) whenever adapter weights are injected into the quantized base model runtime.

## Symptom & Incident Report
During runtime initialization, loading PEFT adapter checkpoints and fusing them into dequantized base model weights yields mismatched numerical outputs compared to standard `llama.cpp` inference outputs.

Investigation indicates three distinct pipeline failures:
1. **Converter Script (`adapter/convert.py`)**: Exported GGUF adapter files fail standard file format validation. Tensor naming conventions and metadata keys for the LoRA alpha and scaling parameters do not conform to GGML specifications.
2. **LoRA GGUF Parser (`gguf_adapter/parser.py`)**: In-memory parsing of LoRA GGUF files miscalculates matrix rank scalings or reconstructs incorrect delta tensor dimensions ($\Delta W = \frac{\alpha}{r} \cdot B \cdot A$), producing shape mismatches or bad scales against target layer weights.
3. **Weight Fusion (`gguf_adapter/dequant.py`)**: Applying dequantized floating-point base matrix transformations alongside reconstructed LoRA delta updates produces high absolute error (`max_abs_err`) against reference GGML matrix fusion.

## Your Task
Fix the adapter conversion pipeline, GGUF LoRA parser, and weight fusion logic so that PEFT adapter checkpoints can be converted, parsed, and correctly applied to dequantized base weights. Finally, implement a regression test suite in `tests/test_regression.py` that verifies adapter application accuracy and catches improper scaling or matrix alignment bugs.
