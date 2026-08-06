Our automated compilation pipeline is exhibiting erratic latency spikes and silent failures during PyTorch model optimization. Diagnostic telemetry shows three distinct issues across compiler execution modes and backends:

First, model snippets deployed with `fullgraph=True` frequently crash or abort, whereas the same snippets under default compilation (`fullgraph=False`) execute with unmonitored graph breaks or succeed cleanly. We lack a diagnostic tool to systematically categorize standard code constructs into `pass`, `graph_break`, or `error` classifications across both modes.

Second, during variable-shape sequence generation, inference latency repeatedly spikes during warm-up runs. The team suspects frequent re-compilations as distinct batch and sequence shapes arrive, but we currently have no way to quantify cumulative compile time penalties or evaluate the threshold where dynamic shape guard generalization becomes net-beneficial over static shape compilation.

Third, benchmarking reports show high initial call times for the Inductor backend, but current metrics combine framework tracing overhead (FX tracing and AOTAutograd) together with backend compilation (C++/Triton codegen), making it impossible to isolate true compiler backend costs.

Build the `compiletracer` package to classify 10 real code snippet patterns, measure dynamic shape recompilation overheads, and isolate backend compilation time.
