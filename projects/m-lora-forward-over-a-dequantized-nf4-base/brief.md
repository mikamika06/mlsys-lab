# QLoRA Adapter Dynamics & Config Safety

Our team is standardizing QLoRA support across low-bit fine-tuning pipelines using `bitsandbytes` and `PEFT`. During recent training runs on dequantized NF4 linear bases, we encountered numerical stability issues and configuration errors.

Specifically, downstream training runs have crashed due to dtype mismatch NaNs when low-precision scales interact with full-precision adapter adapters. Furthermore, the memory planning tool miscalculated parameter budgets because target module expansion rules did not account for partial string matching on quantized layers.

To prevent further pipeline failures, we need a clean, self-contained reference module for QLoRA forward operations, parameter accounting, and config validation.

## Objectives
1. Implement an auto-dequantizing NF4 linear forward pass with optional LoRA adapter addition: $y = x W_{dequant}^T + \frac{\alpha}{r} (x A^T B^T)$. Ensure input and weight precision are unified dynamically during the forward computation.
2. Implement parameter counting for target module specifications under regex-style matching against network layer names.
3. Build a config repair and validation routine that detects potential dynamic precision conflicts (e.g., mismatched compute dynamic types leading to NaNs) and fixes them to a stable execution state.
4. Write a unit test suite in `tests/test_regression.py` that verifies adapter dynamics, parameter budget constraints, and config safety invariants.
