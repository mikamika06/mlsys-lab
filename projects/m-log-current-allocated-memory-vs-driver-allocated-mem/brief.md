When working with the PyTorch MPS backend on Apple Silicon devices, developers often encounter a confusing `OutOfMemoryError`. The device may report that the active tensors are consuming far less memory than the system's unified memory capacity, yet allocations still fail. This occurs because the MPS caching allocator requests memory from the OS in large blocks, which are tracked as `driver_allocated_memory`. When you delete a tensor, that memory is simply returned to the caching allocator, remaining part of the driver's footprint rather than returning to the OS.

Over time, this caching behavior leads to fragmentation. You might have gigabytes of free space residing inside the cache, but it consists of small, non-contiguous chunks. When your model demands a large continuous block, the allocator cannot supply it from the cache and requests a new block from the OS, pushing the total memory footprint past the absolute limit.

In this project, you will investigate this behavior using a simulated MPS caching allocator. You will:
1. Track the divergence between `current_allocated_memory` and `driver_allocated_memory` for specific operational workloads.
2. Reproduce the fragmentation Out-Of-Memory error manually, ensuring `current_allocated_memory` is well below the `recommended_max_memory` limit, and then apply `empty_cache()` to resolve it.
3. Empirically discover the true absolute Out-Of-Memory threshold through a binary search, contrasting it against the soft `recommended_max_memory` boundary.
