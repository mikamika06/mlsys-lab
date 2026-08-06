# Hub-kernel attention backend fallback and performance inspection

During production deployment of models hosted on the Hugging Face Hub, several GPU worker nodes fail during initialization or inference with `ImportError: Flash Attention is not installed`. These instances run on hardware or container environments where `flash-attn` cannot be built or loaded. However, models configured to pull custom attention kernels or backend dispatchers directly from the Hub must remain functional across heterogeneous fleets by falling back gracefully to available AttentionInterface backends (such as PyTorch SDPA or custom manual implementations).

Currently, the custom attention dispatch mechanism crashes when `flash-attn` is absent rather than selecting a supported fallback backend. Furthermore, the dispatch loop lacks a unified training step cost profiler across different attention interfaces, making it impossible to detect performance regressions when a fallback backend is dynamically selected.

You need to implement a robust attention interface router and cost profiling utility under `hf_attn/` that:
1. Dynamically detects available attention backends and resolves Hub model kernel requests down to an executable backend when `flash-attn` is missing.
2. Computes and compares theoretical and execution step costs across backends (`flash`, `sdpa`, `math`) for arbitrary sequence lengths and head configurations.
3. Provides a regression test suite in `tests/test_regression.py` that verifies attention interface dispatch invariants and catches unauthorized fallback overrides.
