Issue Ticket #4092: Fine-tuning pipeline OOMs on 16GB cards and produces corrupted merged weights for deployment.

Our LLM fine-tuning team is attempting to fine-tune a 7B parameter model on a single 16GB GPU instance using QLoRA. Despite enabling 8-bit AdamW and 4-bit base weight quantization, the job crashes during the first backward pass with a CUDA Out of Memory (OOM) exception. The team suspects the VRAM estimate for 8-bit optimizer states is undercounting page table metadata and fails to account for the activation workspace peak during layer-wise backpropagation.

Additionally, when attempting to export merged model checkpoints for post-training inference serving, the merged model outputs complete gibberish. The weight merging routine attempts to combine the low-rank adapter weights A and B directly with the quantized base weights without properly dequantizing the 4-bit blocks using scale factors, leading to severe numerical distortion.

We need a robust memory estimator module to compute exact peak VRAM requirements under paged vs non-paged 8-bit AdamW state allocations, and a clean weight merging and 4-bit quantization pipeline. All components must be validated with regression tests to prevent memory leaks and quantization scale corruption in future releases.
