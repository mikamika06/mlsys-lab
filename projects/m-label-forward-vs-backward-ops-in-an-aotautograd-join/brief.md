# Debugging Joint Graph Traces and Memory Allocations in AOTAutograd

Our model training pipeline experienced severe GPU out-of-memory (OOM) spikes during backpropagation. Initial profiling using TorchDynamo and AOTAutograd trace dumps revealed three core issues in our custom backend graph processing:

1. Operations in the traced joint graph (produced by AOTAutograd) are currently indistinguishable between forward compute and backward gradient propagation. This makes downstream memory profiling and activation-checkpointing passes fail because they cannot identify which nodes belong to which phase.
2. The compiler's recomputation pass is naively recomputing expensive mathematical kernels while saving lightweight elementwise operations, causing both high peak memory usage and unnecessary FLOPs.
3. Graph functionalization is incomplete—in-place mutations (such as `add_`) survive into the traced graph, triggering incorrect derivative calculations and preventing memory safety passes from reordering nodes.

We need you to build a structured analyzer and functionalizer for AOTAutograd joint graph dumps. You will implement node phase labeling, simulate and compare recomputation memory tradeoffs for light versus heavy nodes, and build a graph transformation pass that replaces in-place mutations with pure functional equivalents. Finally, you will write regression tests ensuring the functionalization pass correctly identifies and converts in-place mutations.
