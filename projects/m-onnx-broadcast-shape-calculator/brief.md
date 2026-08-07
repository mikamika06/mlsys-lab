# Ticket: Intermediate tensor shape inference and broadcast triage failure

**Symptom**
Our model optimization and TensorRT conversion pipeline is experiencing non-deterministic runtime crashes during ONNX graph ingestion. When converting deep learning models from PyTorch and JAX exporters, several intermediate layers lack explicit `value_info` shape and type annotations in the exported ONNX protobuf structures.

During graph verification and shape inference passes, our current engine fails to properly compute multi-directional broadcast shapes when tensors have unequal ranks or mix static integer dimensions with symbolic dimension strings (such as `batch_size` or `seq_len`). As a result, elementwise arithmetic nodes (`Add`, `Sub`, `Mul`) either propagate unhandled dimensions downstream or crash with index errors. Furthermore, graph checker passes cannot distinguish between missing intermediate tensor metadata and genuine rank or dimension mismatches, causing broken models to proceed into TensorRT builder allocation where they fail with obscure memory allocation faults.

**Goal**
Implement a dedicated ONNX broadcast shape calculator supporting rank alignment and symbolic dimension resolution, construct a graph-wide shape inference module to rebuild missing `value_info` entries across intermediate nodes, and implement a diagnostic triage checker that catches invalid elementwise broadcasts and rank mismatches before execution.
