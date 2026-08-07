# Debugging AOT Compilation & StableHLO Export Integrity

Our downstream serving system relies on Ahead-Of-Time (AOT) lowering and StableHLO export pipelines for model deployment. During a recent production rollout, several edge devices reported unexpected numeric drift and subtle runtime mismatches when executing serialized StableHLO artifacts compared to standard `jax.jit` execution.

We suspect that during the transformation pipeline, the lowered text representation from `.lower().as_text()` and the serialized byte payloads exported via `jax.export.export` are encountering subtle discrepancies or op-level transformations that go undetected until execution time.

Your task is to implement an inspection and verification utility package (`jaxinspect`) that parses recorded lowered StableHLO representations, extracts IR statistics, and validates numerical consistency across compilation artifacts and serialized process boundaries.

## Required Implementation

Implement the functions across the package as follows:

1. **`jaxinspect.verify`**:
   - `verify_compile_vs_jit(aot_outputs: list[dict], jit_outputs: list[dict]) -> dict`: Given recorded tensor outputs from `.lower().compile()` calls and standard `jax.jit()` execution, compare them entrywise. Calculate the maximum absolute error across all outputs and return a summary containing `max_abs_err` and a boolean `is_close` indicator (true if `max_abs_err <= 1e-5`).

2. **`jaxinspect.ir`**:
   - `analyze_stablehlo_ir(ir_text: str) -> dict`: Parse a StableHLO text representation (as returned by JAX's `.lower().as_text()`). Extract and count all `stablehlo.*` operations. Return a dictionary with key `"op_counts"` mapping unique opcode names (e.g. `"stablehlo.dot_general"`) to their integer counts, and `"unique_ops"` containing a sorted list of unique opcode names.

3. **`jaxinspect.export`**:
   - `verify_serialized_numerics(original_outputs: list[dict], deserialized_outputs: list[dict], rtol: float = 1e-5, atol: float = 1e-5) -> dict`: Verify numerical equivalence between original execution outputs and outputs from an artifact serialized via `jax.export.export` and executed in a fresh process environment. Return a dictionary with `max_abs_err` and `matches` (true if all outputs match within specified tolerances).

4. **`tests/test_regression.py`**:
   - Write regression tests that validate IR parsing invariants and numeric precision checks. Your tests must catch broken implementations that drop IR op attributes or incorrectly compute maximum absolute error across multi-array outputs.
