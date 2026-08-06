# GGUF Model Conversion and Tensor Mapping Utilities

When converting HuggingFace transformer models into GGUF format for llama.cpp execution, downstream loading and inference fail with missing tensor errors and corrupted output.

Specifically, the model loader reports `unknown tensor name` errors when encountering standard HuggingFace keys such as `model.layers.0.self_attn.q_proj.weight`. When tensor names are manually mapped, attention key/query projections produce nonsense activations because HuggingFace and GGUF handle Rotary Position Embedding (RoPE) tensor dimensions using different permutation layouts. In addition, inspection tools and quantization workflows attempting to select tensors via regular expressions (e.g. `--tensor-type`) fail to accurately match specified layer blocks or weight categories.

We need a dedicated conversion module `ggufmap` that correctly translates HuggingFace layer and tensor naming conventions to GGUF standards, reverses GGUF RoPE dimension permutations to align weights back to standard HuggingFace layout, and matches tensor keys against python regex patterns.
