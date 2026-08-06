You are building tools to analyze logs from Triton kernels executed in interpreter mode (`TRITON_INTERPRET=1`). When debugging low-level ML systems, the Triton interpreter provides two crucial pieces of information: standard output from `tl.device_print` and precise `program_id` coordinates when an exception (like an out-of-bounds memory access due to a missing mask) occurs.

Additionally, the interpreter has documented limitations with `bfloat16` data types (e.g., printing full precision float32 values instead of native bf16 representations), which we need to detect in the logs.

Your task is to build the parsing utilities for these interpreter outputs:
1. **Log per-block accumulator values:** Implement `parse_device_print` to extract the block coordinates and accumulator values from the interpreter's stdout stream.
2. **Pinpoint the offending program_id:** Implement `extract_program_id` to parse interpreter-raised masking exception tracebacks and return the exact `(x, y, z)` block coordinate that crashed.
3. **Safety net:** Write robust regression tests for your parser. We will run your tests against a deliberately broken regex engine that fails on multi-digit coordinates to ensure your tests catch the fault.
