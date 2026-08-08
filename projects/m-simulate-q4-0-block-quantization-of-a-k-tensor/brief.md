We are observing high memory consumption and latency overhead during the local deployment of large language models. During heavy decoding phases with long context lengths, the memory footprint of the uncompressed KV cache becomes the primary bottleneck for concurrency.

To alleviate this, we are migrating to the Q4_0 block quantization format for the K tensors. This format groups tensor values into blocks of 32, representing each block with a single float32 scaling factor and packing the values into 4-bit nibbles to drastically save memory bandwidth.

However, we are noticing numerical discrepancies and severe divergence in attention scores compared to the unquantized baseline. We suspect the current implementation is incorrectly unpacking the nibbles or miscalculating the block scale factors.

Your task is to implement a robust numpy-based simulation of Q4_0 block quantization and dequantization. You must partition tensors into blocks of 32, compute the correct scaling factors (using `max_val / -8.0`), pack the values into 4-bit nibbles, and accurately reconstruct the dequantized tensor. You must ensure that the maximum absolute error against the original tensor remains strictly within our bounds. Finally, write a regression test suite that validates the roundtrip bounds and guards against scale computation faults.
