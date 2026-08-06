An internal debugging dashboard is analyzing the low-level tracing mechanics of JAX programs, specifically focusing on how gradients and jit-compilation interact when combined in complex ML systems. Engineers have reported a persistent issue when running codebases that wrap functions with both `jax.jit` and `jax.grad`: the system occasionally reports unexpected equation counts or encounters internal JAX errors related to leaked tracers.

Specifically, when a closure captures mutable state or when multiple transformations are applied in sequence, the underlying jaxpr representation grows unexpected equations, and certain patterns of local state capture trigger tracer lifetime violations (leaked-tracer errors) during compilation and tracing phases.

Your task is to build a utility package under the `jaxpr_tools` namespace that:
1. Accurately counts equations and extracts unique primitives from a grad+jit-composed jaxpr.
2. Identifies and safely reproduces or avoids the leaked-tracer bug caused by mutable closure lists during tracing.
3. Implements a robust test suite that catches regressions when tracer extraction or closure handling is altered.
