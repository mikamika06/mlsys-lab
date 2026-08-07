# Symbolic Shape Propagation by Hand

Our ONNX runtime export pipeline currently crashes when processing dynamic dynamic-batch graphs. While standard shape inference propagates fixed tensor dimensions, dynamic shapes require symbolic shape propagation across complex DAGs (e.g., MatMul broadcasting, Reshape with -1, and Concatenation).

Currently, models with unknown sequence lengths or dynamic batch sizes crash deep in the execution engine because shape signatures are either uninitialized, improperly broadcasted, or lost after dynamic operations. We need a deterministic symbolic shape propagation engine that infers symbolic expressions across node operations, identifies the exact node where inference fails due to missing symbols or rank mismatches, and tracks known-shape coverage before and after propagation.

Fix this issue by implementing symbolic shape inference, first-node failure tracking, and shape coverage analysis.
