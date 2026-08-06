We are wiring up a custom FlashAttention engine that needs to run across different accelerator backends without crashing the host process during initialization. In production, certain hardware backends or driver versions might fail during import or capability probing due to missing symbols, unavailable shared libraries, or mismatched CUDA architectures. If the initialization routine performs a naive import or an unsafe hardware check at top-level module load time, the entire serving cluster crashes before fallback logic can engage.

Your task is to implement a robust backend ladder selector and verification suite in the `flashsel` package. 

First, you need to implement an import-safe capability probe that safely detects available backends without triggering unhandled exceptions or fatal runtime aborts when a driver or library is absent. 

Second, you must build a prioritized backend ladder selector that queries the probed capabilities, respects user-supplied environment overrides or preferences, and gracefully falls back through the ladder until it finds a fully functional, compatible implementation. 

Finally, you need to implement a cross-backend equivalence test in `tests/test_regression.py` that verifies output parity between the primary selected backend and a reference NumPy implementation under identical inputs, ensuring that fallbacks do not silently produce corrupted attention scores or incorrect scaling factors.
