# Ticket: Training Out-Of-Memory Error on Step 300

## Symptom
We are running fine-tuning on a LLaMA 7B model quantized to 4 bits using QLoRA. The training script runs stably for the first 200 steps, but consistently crashes with a CUDA Out-Of-Memory (OOM) error around the 300th step. Our hardware constraint is strictly 24 GB of VRAM. We must complete the training run to the final step without reducing the effective batch size.

## Context
During initial debugging, we noticed that memory usage creeps upward steadily past step 200, eventually exceeding the 24 GB limit. Since we are using QLoRA, the base model weights are frozen in 4-bit, and only the adapter parameters and optimizer states for those adapters should consume dynamic memory. However, intermediate activations, uncollected tensor references, or unoptimized optimizer state precisions might be causing ballooning memory allocations.

We need to systematically analyze memory components, trace what grows across iterations, apply gradient checkpointing, move optimizer states to 8-bit precision, and ensure convergence and effective batch size remain fully preserved through to the end of the run.
