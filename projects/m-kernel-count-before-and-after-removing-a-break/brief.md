We are debugging a performance regression in our PyTorch 2.x training loop. When inspecting the output from `TORCH_LOGS="output_code" python train.py`, we suspect that a graph break is preventing `torch.compile` from fully fusing our pointwise operations into a single Triton kernel.

Your task is to build a tooling layer to analyze Inductor's output and prove the memory bandwidth savings of fusion:

1. **Parse the output code dump**: PyTorch Inductor generates Python modules containing Triton kernel definitions (e.g., `def triton_poi_fused_add_0(in_ptr0, ...):`). Write a parser to find these definitions, count them, and extract the pointer arguments they take (ignoring structural arguments like `xnumel` and `XBLOCK`).
2. **Calculate bytes moved**: To justify removing the graph break, implement a bandwidth calculator for a chain of pointwise ops. If `fused=False`, every operation reads its inputs from DRAM and writes its output to DRAM. If `fused=True`, intermediate tensors live entirely in GPU registers; only the true inputs of the chain and the final true outputs touch DRAM.
3. **Safety net**: Write a regression test verifying that fusion strictly reduces bytes moved for a multi-operation chain. Our harness will simulate a "broken" `calc_bytes` function that incorrectly falls back to unfused counts—your test must fail when this happens.
