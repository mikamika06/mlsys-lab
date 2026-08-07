We are investigating a performance regression in our model serving infrastructure. The median inference time jumped by over 15% following a recent deployment, and the variance has significantly increased. We pulled the PyTorch profiler Chrome traces (`.json` files) from the production endpoints to investigate the bottleneck.

Looking at the PyTorch profiler's standard `key_averages()` output, we can clearly see that the `aten::mm` and `aten::bmm` operators are taking up a huge portion of the total execution time. However, this high-level summary is not actionable. We have a highly dynamic workload with highly variable sequence lengths, batch sizes, and sliding window configurations.

The standard `key_averages()` merges all `aten::mm` calls together, regardless of their input shapes. We suspect that only a few specific matrix multiplication shapes (for example, the prefill phase with extremely large contexts) are causing the bottleneck, while the decoding shapes remain efficient.

We need a custom trace analyzer. It must read the raw Chrome Trace Event Format JSON, extract the `Input Dims` from the operator arguments, and calculate the top-k operators grouped not just by operator name, but by their specific input shapes. This exact breakdown will let us pinpoint the exact tensor dimensions we need to optimize or shard differently.
