# Channels Last & Host-side Overlap

We recently migrated our PyTorch ResNet training to use `channels_last` memory format (NHWC) to take advantage of Tensor Core speedups on our GPUs. To keep the GPU fed, we added `non_blocking=True` to our `.to(device)` calls to overlap the host-to-device transfers with the computation of the previous batch.

However, our profiler shows a dismal timeline:
1. The CPU blocks entirely during the H2D transfer.
2. The GPU sits idle waiting for data, with absolutely no overlap!

Also, our C++ data augmentations are producing manual tensor structures. Even though they output shapes of exactly `(N, C, H, W)`, PyTorch complains they are not `channels_last` when we pass them to `conv2d`.

I need you to build two things to debug our pipeline:
1. A layout inspector (`layout.strides`) that calculates the correct strides for a contiguous NHWC tensor (given its NCHW shape) and strictly verifies if a tensor is formatted correctly.
2. A pipeline simulator (`layout.pipeline`) that calculates the steady-state batch time. You'll need to model how `pin_memory` dictates whether the transfer is synchronous (blocking the CPU) or asynchronous (allowing overlap).
