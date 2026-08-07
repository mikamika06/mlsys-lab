# Trace vs Export: Which Branch Got Baked?

We are seeing silent failures in our edge model export pipeline when attempting to convert control-flow heavy PyTorch models. PyTorch's trace-based engine (`torch.jit.trace`) silently bakes in branch decisions made during tracing, while graph capture (`torch.export.export`) either rejects invalid dynamic control flow or requires explicit dynamic shapes and guards.

In production, models traced with conditional branches (e.g., dynamic thresholding, conditional sequence routing) return incorrect outputs when fed inputs that should trigger alternate control flow paths. Additionally, exporting dynamic models without minimal dynamic shape bounds causes runtime graph reconstruction failures on edge runtimes.

Your task is to implement an inspection and capture tool under `gcapture/` to analyze graph branches, deduce dynamic shape constraints, and profile ATen operator distributions across capture methods.

## Objective
Implement three core submodules in `gcapture/`:
1. `branch.py`: Trace and export a module with control flow to determine which conditional branch gets hardcoded by `torch.jit.trace` versus retained or guarded under `torch.export`.
2. `dynshape.py`: Diagnose `torch.export` constraint violations given out-of-spec test inputs, and construct a minimal `torch.export.dynamic_shapes` specification.
3. `histogram.py`: Parse exported graph structures down to their core ATen decomposed operations, computing an accurate operator frequency histogram for runtime target validation.
