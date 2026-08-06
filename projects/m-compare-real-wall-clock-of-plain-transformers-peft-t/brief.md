# Benchmarking Framework Efficiency: PyTorch/PEFT/TRL vs. MLX on Apple Silicon

## Symptom
Your team is standardizing fine-tuning workflows across a fleet of Apple Silicon workstations. An engineer onboarded a new task using the default PyTorch stack (`transformers` + `peft` + TRL `SFTTrainer`), while another prototype running on `mlx_lm.lora` reportedly finishes fine-tuning loops faster with significantly lower memory usage.

Because both pipelines are producing fine-tuned models for downstream export, management wants hard metrics to justify whether to transition all local training pipelines on Apple Silicon to MLX. However, ad-hoc terminal output comparisons across machines have led to conflicting claims regarding execution latency, peak memory allocation, and loss convergence.

## Objective
Build a automated framework comparison runner to measure and audit performance metrics between PyTorch-based PEFT and native MLX fine-tuning implementations:

1. **Execution Latency Profiling:** Profile total wall-clock runtime for fine-tuning loops across PyTorch and MLX, computing relative speedup ratios and tracking per-iteration step times.
2. **Peak Memory Footprint Auditing:** Track peak Unified Memory allocation and record memory deltas between PyTorch's MPS / unified memory manager and MLX's active allocation pool.
3. **Loss Convergence Verification:** Audit training loss curves across both frameworks to confirm that LoRA adapters trained via PyTorch PEFT and MLX maintain loss parity over equivalent optimization trajectories.
4. **Automated Safeguard Tests:** Implement test assertions in `tests/test_regression.py` that systematically identify and reject distorted benchmark comparisons (such as mismatched hyperparameter schedules, un-synchronized timer calls, or bogus loss trajectories).
