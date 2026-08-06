Our automated LLaVA GGUF conversion pipeline is failing. The inference engine spits out errors like `tensor v.patch_embd.weight not found` when loading the `mmproj` sidecar file we produce.

Looking at the weights dictionary loaded from the HuggingFace checkpoints, we have keys like `vision_model.embeddings.patch_embedding.weight`, `multi_modal_projector.linear_1.weight`, and `vision_model.encoder.layers.0.self_attn.k_proj.weight`. The inference engine, however, expects the standard GGUF nomenclature: `v.patch_embd.weight`, `mm.proj.1.weight`, and `v.blk.0.attn_k.weight`.

It seems we are dumping the raw PyTorch names directly into the GGUF file instead of translating them to the canonical vision encoder layout. This causes the ggml backend to fail because it has no idea what `vision_model.embeddings...` means.

We need a function `map_tensors(raw_names)` that takes a list of raw string keys and returns a dictionary mapping them to the proper GGUF names. The mapping should handle the base vision embeddings, the post-layernorm, the multimodal projector itself, and all of the encoder blocks (including the self-attention projections, the layer norms, and the MLP components `fc1` and `fc2`). Make sure you handle `.bias` where applicable, passing it through identically in the GGUF name.
