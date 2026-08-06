# Ticket #7419: Non-deterministic gradients and finite difference mismatch in fused Triton backward kernel

**Status:** Open  
**Component:** Custom Kernel Verification & Backward Pass (`rw3-triton-real`)  
**Priority:** High  

## Symptom Description
During production training of our custom fused activation transformer block, we observed severe loss divergence and non-deterministic training runs across identical hardware configurations using fixed random seeds.

When isolating the issue to individual layers, the forward pass kernel produces exact outputs that match our reference PyTorch implementation down to floating-point machine epsilon. However, the backward pass kernel exhibits two critical failures:

1. Invoking the backward pass kernel twice on identical input tensors yields non-deterministic gradient tensors `dx` at specific memory locations.
2. When validating the backward kernel's output gradients against a central finite difference baseline, certain input indices exhibit massive numerical discrepancies while others match perfectly.

Initial telemetry suggests that gradient inconsistency only appears on tensor shapes and index mappings where multiple output elements gather from the same input index in the forward pass. We need a clean backward kernel implementation for the fused elementwise operation verified against finite differences, a classifier function to determine when atomic operations are mandatory based on index mappings, and a determinism diagnostic comparing atomic versus non-atomic gradient accumulation.
