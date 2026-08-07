Our long-context models are occasionally hitting out-of-memory (OOM) errors during training. The infrastructure team just added two Sequence Parallelism (SP) backends: Ulysses-SP and Ring-SP. We need to evaluate them and verify their implementations are correct.

First, we collected DeepSpeed communication logs to verify the Ulysses-SP implementation. We suspect the `all_to_all` communication volume is larger than it should be. Write a verification function in `sp_comm/comm_log.py` that computes the theoretical expected `all_to_all` send volume and compares it to the sum in the logs.
For a single layer, the total expected `all_to_all` data sent per GPU is the sum of QKV and Attention Output exchanges:
- QKV: `((P - 1) / P) * (3 * S * b * h / P) * bytes_per_elem` bytes.
- Out: `((P - 1) / P) * (S * b * h / P) * bytes_per_elem` bytes.
Return the relative error `abs(actual - expected) / expected` across all `L` layers. If `expected` is 0, return 0.0 if actual is 0, else return `float('inf')`.

Second, we need a memory budget planner in `sp_comm/mem_budget.py`. Given a fixed `mem_budget` (in bytes), compute the maximum sequence length `S` for three strategies: `dense` (no SP, effectively P=1), `ulysses`, and `ring`.
Memory components:
- Static memory (weights + opt): `16 * L * 12 * (h ** 2)`
- Activations (per layer): `34 * b * S_per_gpu * h * bytes_per_elem`, where `S_per_gpu` is `S` for dense, and `S / P` for SP.
- SP Buffer memory: `dense` uses 0. `ulysses` uses `3 * b * h * bytes_per_elem / P * S`. `ring` uses `2 * b * h * bytes_per_elem / P * S`.

If `mem_budget` is less than the static memory, return 0 for all. Otherwise, solve for the maximum `S` using float division for the memory coefficients and `int(...)` to truncate the final sequence lengths. Return a dictionary with keys `"dense"`, `"ulysses"`, `"ring"`.
