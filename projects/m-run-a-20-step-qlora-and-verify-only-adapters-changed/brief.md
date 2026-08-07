QLoRA fine-tuning significantly reduces memory requirements by combining frozen, quantized base weights with trainable low-rank adapters (LoRA). The core idea is that we dequantize the base model's weights on the fly during the forward pass, and only propagate gradients back to the lightweight adapter weights.

A common pitfall when integrating LoRA manually is failing to freeze the base model weights, causing massive parameter updates that destroy the memory benefits of QLoRA or crash the machine out of memory (OOM).

Your task is to:
1. Implement a `LinearQLoRA` layer in `qlora/layer.py` that dequantizes a simulated 8-bit weight matrix and applies a LoRA adapter on top.
2. Write the backpropagation logic and a 20-step training loop in `qlora/train.py` that updates *only* the adapter weights (`lora_A` and `lora_B`).
3. Write a regression test in `tests/test_regression.py` that explicitly trains the layer and asserts that the base weights and scales remain bit-for-bit identical, while the adapter weights change.

We provide the `__init__` constructor for the layer with a mock int8 quantization scheme.
