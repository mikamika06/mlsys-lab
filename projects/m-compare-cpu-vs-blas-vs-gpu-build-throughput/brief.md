# Ticket: Build Throughput Discrepancy and CMake Flag Drift in Llama.cpp Backends

## Symptom
The automated continuous integration and release engineering pipeline for our low-level GGUF and llama.cpp distribution is experiencing severe observability gaps and configuration drift. Engineers running local compilation benchmarks report conflicting build throughput numbers when comparing pure CPU targets, OpenBLAS-accelerated builds, and CUDA/GPU builds. Specifically, build times and token generation throughput metrics fluctuate wildly without a clear deterministic baseline, making performance regression tracking nearly impossible.

Furthermore, during a recent codebase cleanup intended to strip out legacy quantization block variants—specifically removing the experimental `Q4_0_4_4` layout implementation—the resulting binary and model container file sizes stubbornly remained identical down to the exact byte. Reviewers are unable to verify whether the removal was dead code, if storage footprint calculations are broken, or if an implicit fallback alias is silently masking the code deletion.

Finally, backend-specific compilation scripts are plagued by inconsistent CMake flag enumerations. Different scripts invoke target accelerators using mismatched definitions (`GGML_CUDA`, `GGML_OPENBLAS`, `LLAMA_CUBLAS`, etc.), causing silent build degradation where hardware acceleration is omitted without throwing an explicit configuration error.

We need a robust, deterministic Python analysis framework within the build subsystem to parse build throughput ratios, mathematically explain the file size invariance of the `Q4_0_4_4` removal, and systematically validate complete CMake flag matrices for all target backends.
