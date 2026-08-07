Ticket: INC-84920 - Silent CPU Fallback and Unverified Node Placement in ORT Serving Nodes

Our production inference service uses ONNX Runtime with CUDA and TensorRT execution providers. During recent rolling deployments across heterogeneous GPU nodes, several worker instances experienced severe throughput degradation—latency jumped from 14ms to nearly 900ms per batch. Investigation revealed that when certain GPU worker nodes had mismatched CUDA/cuDNN shared library versions or missing provider dependencies, ONNX Runtime silently fell back to `CPUExecutionProvider` without raising an error or failing the container health check.

Additionally, even when `CUDAExecutionProvider` is active, certain unsupported subgraphs or operator variants silently execute on CPU, leading to partial provider fallback that goes unnoticed in standard metrics.

We need to implement an `ortpreflight` tool suite that:
1. Performs strict provider preflight validation against an explicit CUDA/cuDNN version compatibility oracle before creating ORT sessions, failing fast if the requested primary provider cannot run.
2. Inspects ORT verbose initialization logs to extract precise node-to-Execution-Provider assignments and compute operator placement distribution ratios.
3. Includes a regression test suite catching silent CPU fallback bugs before code reaches staging.
