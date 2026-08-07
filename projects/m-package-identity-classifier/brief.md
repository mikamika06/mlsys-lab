# Symptom: Intermittent runtime crashes and incorrect dispatch logic in FlashAttention kernels

We are experiencing nondeterministic execution failures and wrong kernel selections during FlashAttention invocations in multi-GPU production environments. On systems featuring mixed device generations and varying CUDA toolchains, models occasionally launch fallback execution paths or crash with architecture mismatch errors during tensor operations.

After preliminary inspection of the system logs, the issue appears tied to runtime package detection and hardware gating. The environment classifier incorrectly identifies installed package capabilities or misjudges host GPU capability profiles, leading to improper kernel choices or attempts to invoke unavailable optimized primitives.

Your task is to build a robust, deterministic package-identity and hardware-capability classifier that accurately maps installed library builds, CUDA capability metadata, and runtime features to valid operational dispatch targets, preventing invalid kernel invocations.
