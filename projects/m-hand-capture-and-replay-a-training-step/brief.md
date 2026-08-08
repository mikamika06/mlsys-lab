# Debugging PyTorch CUDA Graph Capture and Memory Overwrites

During static CUDA Graph capture and replay in low-level training loops, operators occasionally produce unexpected numeric outputs or zero gradients. Profiling reveals that memory addresses allocated for static CUDA Graph output buffers are being inadvertently shared or overwritten across iterations or subsequent kernel executions.

In CUDA Graph capture mode (`torch.cuda.make_graphed_callables` or manual stream capture), operations are recorded without immediate execution. Static output tensors are assigned fixed memory addresses in the CUDA Graph memory pool. When non-capture-safe ops (like in-place operations or improper static buffer reuse) occur during the training step, subsequent graph replay iterations overwrite active tensor outputs or corrupt intermediate activation buffers.

Your task is to analyze CUDA graph capture traces, diagnose whether operations are capture-safe, hard errors, or silently wrong, and implement memory allocation and buffer management strategies to fix tensor output overwrites during CUDA graph capture and replay.
