# Symptom Report: Unpredictable Graph Breaks During TorchDynamo Bytecode Tracing

Our machine learning operations team has encountered significant latency variance and compilation overhead when running PyTorch models with TorchDynamo JIT compilation (`torch.compile`). Certain model blocks compile cleanly into a single optimized graph, while other structurally similar snippets trigger frequent graph breaks, forcing the tracer to split execution across multiple fallback bytecode frames.

Currently, engineers lack visibility into how many sub-graphs are generated before submitting functions to full compilation. When profiling model code snippets with nested conditional constructs (`if`/`else` control flow) or Python side-effects such as logging and printing, there is no quick way to compute `graph_break_count` or predict the resulting `graph_count` (`graph_break_count + 1`).

We need a lightweight static tracer module in `break_analyzer` that parses AST structures from model code snippets, identifies graph break triggers, ranks candidate model snippets by break count, and provides an explainer function to verify graph count predictions on nested-if model functions.
