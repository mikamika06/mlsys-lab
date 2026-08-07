# Transpose Overhead from Edge Layout Mismatch

A high-throughput video analytics engine experiencing dropped frames after model export to an edge accelerator was traced back to runtime memory copy bottlenecks. The host application captures frames in HWC (height, width, channels) contiguous memory layout, while the exported model graph expects NCHW layout for spatial convolutions.

Currently, preprocessing is split between host application code and exported graph nodes, introducing repeated memory transposes and redundant memory allocations on every inference call. Performance telemetry shows that memory copy operations between Host RAM and Accelerator Memory consume a significant fraction of the total per-frame inference budget.

Your task is to analyze the layout mismatch, unify the input pipeline strategy, and implement a validation checker to verify equivalence between application-side and in-graph preprocessing.

## Objective
1. Measure layout transpose overhead and convert between `NHWC` and `NCHW` efficiently, isolating layout transformations from image scaling and normalization.
2. Build dual-path pipelines (app-side layout conversion vs. in-graph layout transformation) and calculate memory footprint differences, transfer overheads, and total execution latency across batch configurations.
3. Construct a numerical equivalence checker and regression suite to verify that in-graph and app-side preprocessing produce identical outputs within floating-point tolerances.
