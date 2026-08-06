# Investigating Parameter Count Miscalculations and Adapter Fusion Inconsistencies in MLX LoRA Fine-Tuning

We recently noticed discrepancies in our MLX fine-tuning monitoring pipeline. The trainable parameter count reported by our initial theoretical formulas for `mlx_lm.lora` does not match the actual trainable parameter counts observed during fine-tuning runs. This introduces budget allocation errors when scaling multi-adapter jobs.

Additionally, our serving pipeline fuses adapter weights into the base model using `mlx_lm.fuse` before deployment. However, several downstream latency and accuracy verification checks reported numerical divergence between the fused base model and the pre-fuse adapter-attached model.

Finally, our training resource estimation tool requires accurate relative benchmarks comparing standard LoRA against DoRA (`dora`) fine-tuning in MLX across memory usage and training wall-clock time per step.

To resolve these operational issues, we need a unified audit module that:
1. Computes exact trainable parameter counts for target MLX layer specifications (accounting for LoRA vs. DoRA scale vectors and rank dimensions) and contrasts them against real `mlx_lm` layer structures.
2. Formally verifies weight identity between a fused `mlx_lm` model and its unfused adapter equivalent.
3. Profiles real memory allocations and step times across LoRA and DoRA training passes to ensure parameter optimization choices remain within cluster memory limits.

Complete the implementation in `mlx_lora_audit/params.py`, `mlx_lora_audit/fuse.py`, and `mlx_lora_audit/bench.py`, and write an invariant regression test in `tests/test_regression.py`.
