# Storage and Fleet Deployment Matrix for TensorRT Engines

Our ML platform team is seeing deployment friction when deploying TensorRT engines across heterogeneous cloud environments.
Specifically, storage footprint scales exponentially when generating pre-built TRT engines across multiple CUDA container patch versions, GPU microarchitectures, and runtime configurations. Additionally, deployment rollouts occasionally stall because engine artifacts built inside updated containers report minor patch mismatches, while engines compiled with full version-compatibility flags inflate binary size beyond storage quotas.

We need a unified library in `artifact_matrix/` to plan, evaluate, and regression-test our TensorRT deployment matrix:

1. **Artifact Matrix Size Planning (`artifact_matrix/planner.py`)**:
   Implement `calculate_matrix_size(architectures, trt_versions, runtime_modes, engine_base_mb, compatibility_bloat_factors)` to calculate the exact total storage footprint (in MB) for an engine artifact matrix across architectures, TensorRT versions, and runtime modes (e.g., standard vs. version-compatible).

2. **Container Patch Mismatch Resolution (`artifact_matrix/container_patch.py`)**:
   Implement `resolve_container_patch(container_version, engine_version, patch_policy)` to resolve compatibility between container runtime patches and engine metadata according to team rules (`strict`, `allow_patch_drift`, `auto_patch_alias`).

3. **Version-Compatible Engine Storage Cost (`artifact_matrix/cost_model.py`)**:
   Implement `estimate_vc_engine_cost(base_engine_mb, enable_vc, refit_enabled, lean_runtime)` to compute the final engine size and cost delta when enabling TensorRT Version Compatibility (VC) and Lean Runtime options.

4. **Safety Net (`tests/test_regression.py`)**:
   Write a suite of pytest/unittest functions in `tests/test_regression.py` that verify container patch resolution and matrix size accounting. The harness will inject a broken patch resolver to verify your regression suite actively catches illegal patch alias drift.
