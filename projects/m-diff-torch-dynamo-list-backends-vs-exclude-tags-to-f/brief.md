# Ticket: Compiler Discovery and Overhead Profiling Utility Fails on Custom Backends

## Symptom Description

The internal model optimization pipeline is currently experiencing inconsistencies when trying to discover cutting-edge or experimental compiler options within the runtime environment. Engineers report that standard backend enumeration calls omit several registered compilation targets that are tagged as experimental or debug-level, preventing automated fallback scripts from evaluating them.

Furthermore, when attempting to profile the overhead of custom compilation pathways versus the standard default production compiler (Inductor), performance monitoring scripts either hang, throw type errors, or return inverted ratio metrics. Specifically, evaluating a lightweight no-op custom backend alongside Inductor on standard benchmark layers produces erratic timing ratios, making it impossible to determine whether the custom integration introduces unacceptable compilation latency or execution overhead during dynamic graph tracing.

We need a dedicated sub-package that robustly separates standard backends from experimental ones using proper tag exclusion filtering, and cleanly computes reliable compile-time and run-time performance ratios for custom no-op backends against Inductor under controlled, deterministic conditions. This sub-package must integrate smoothly with existing test suites and allow strict regression verification to ensure profiling invariants remain unbroken during future updates.
