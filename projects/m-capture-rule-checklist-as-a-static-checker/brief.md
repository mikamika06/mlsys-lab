We've noticed that our training pipeline occasionally crashes or experiences severe performance degradation when CUDA Graph capture is enabled via PyTorch's `mode='reduce-overhead'`. Engineers have been manually inspecting dynamic Python trace logs and PyTorch FX graphs before enabling capture, but non-compliant operations still slip into production runs.

Common issues include host-device synchronization calls, dynamic memory allocation operations, CPU-CUDA data transfers, and non-deterministic kernel invocations during the captured forward/backward pass.

We need an automated static analysis system to lint target PyTorch FX graphs against CUDA Graph capture rules before graph capture begins.

Your task is to build a static checker module `graph_checker` with the following components:
1. `graph_checker/checker.py`: Implement `check_graph_violations(gm: torch.fx.GraphModule) -> list[dict]` to inspect an FX `GraphModule` and return a list of rule violations. Each violation record must be a dict containing keys `"node"`, `"rule"`, and `"severity"`. You need to check for rules:
   - `SYNC_OP`: Host-device syncs (e.g., `.item()`, `.nonzero()`, `torch.cuda.synchronize`).
   - `MALLOC_OP`: Explicit dynamic allocations or size reallocations during graph run (e.g., `torch.empty`, `torch.zeros` inside the forward pass).
   - `H2D_TRANSFER`: Host-to-Device or Device-to-Host copies inside the graph (`.to('cuda')`, `.cpu()`).
2. `graph_checker/optimizer.py`: Implement `suggest_safe_transforms(gm: torch.fx.GraphModule) -> torch.fx.GraphModule` which transforms simple violation patterns into CUDA Graph compliant alternatives (e.g. replacing inline `torch.empty` allocations with pre-allocated buffer nodes or constant fills where safe).
3. `tests/test_regression.py`: Write tests validating that your static checker catches illegal ops and verifies transformed graphs pass inspection without raising violations.
