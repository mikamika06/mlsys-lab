# Ticket: Production TRT Engine Build Failures from Frozen Dynamic Axes and Graph Bloat

## Symptom
Our TensorRT engine compilation pipeline has started rejecting exported ONNX graph artifacts in staging. Models that were originally exported with dynamic batch sizes and dynamic sequence lengths are suddenly failing during engine construction with runtime shape mismatch errors. Downstream deployment workers report that engines compiled from these graphs crash when served requests with variable batch sizes, throwing invalid dimensions errors because previously symbolic dimensions (such as `batch` or `sequence`) have been hardcoded to static integers like `1` or `128`.

Initial analysis of the optimization telemetry indicates that constant-folding passes during graph simplification evaluate subgraphs that derive shape metadata, inadvertently collapsing dynamic shape expressions into static constants. Additionally, these premature folding passes leave orphaned initializers and disconnected subgraphs behind in the ONNX graph structure. This unnecessary payload increases serialization overhead and slows down graph optimization passes during build time.

We need a lightweight diagnostic utility and cleanup package to scan ONNX graphs, pinpoint constant-fold nodes that freeze dynamic axes, sweep dead nodes and orphan initializers, and compute simplification payoff metrics to safeguard our deployment pipeline.
