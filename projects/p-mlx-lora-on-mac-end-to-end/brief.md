# Ticket: LoRA End-to-End Pipeline on Apple Silicon

## Symptom
When attempting to domain-adapt a base large language model locally on Apple Silicon hardware without relying on cloud infrastructure, engineers face a fragmented workflow. Raw domain text cannot be seamlessly ingested by local MLX training loops without custom tokenization and batch structuring. Furthermore, once adapters are trained, engineers struggle with extracting loss convergence metrics, merging LoRA adapter weights back into the baseline model weights cleanly, and converting formats.

Additionally, serving the unoptimized fine-tuned model results in excessive memory footprint and latency spikes that exceed unified memory limits on local M-series hardware. Without a robust quantization and serving layer wrapper, client integration tests fail under load or exhibit incorrect token generation. Finally, verifying quality regressions before and after fine-tuning lacks a rigorous automated evaluation harness, leaving deployment confidence entirely unquantified.

## Goal
Implement a complete end-to-end local fine-tuning and serving pipeline module in `lora_pipe/engine.py` that handles data preparation, LoRA training simulation with loss tracking, weight merging, model quantization, local inference server execution, and rigorous test coverage with regression validation.
