# Distributed Training Failure and Communication Overhead Under ZeRO Offload

## Symptom
Distributed LLM training jobs across multi-node GPU clusters are failing during job initialization or experiencing extreme communication bottlenecks.

When attempting to scale model capacity using ZeRO-Offload and ZeRO-Infinity strategies, cluster launch scripts crash with CUDA out-of-memory errors on GPU devices even when huge amounts of system CPU RAM remain free. The current planner incorrectly calculates the maximum trainable parameter bound, failing to balance parameter partitioning, activation buffers, and CPU-offloaded optimizer states.

Furthermore, network profiling logs reveal that parameter gather volume during forward and backward passes is bottlenecking the Inter-Node interconnect. Although ZeRO++ optimizations (specifically hierarchical partitioning hpZ and quantized all-gather qgZ) are enabled in the configuration, all-gather transfers remain uncompressed, and inter-node communication traffic fails to demonstrate the expected fourfold volume reduction compared to standard ZeRO-3. Additionally, custom block quantization routines for parameter tensors exhibit shape mismatches during dequantization and produce excessive relative reconstruction error.

We need utility modules to accurately calculate maximum trainable parameter bounds under CPU offload constraints, perform loss-bounded INT8 block quantization for hpZ parameter gathers, and verify communication volume reduction invariants via regression testing.
