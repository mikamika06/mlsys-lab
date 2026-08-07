# Ticket: Model Quantization Under Accuracy Floor

We have deployed a dense baseline model that consumes excessive memory during inference, doubling our operational costs and pushing latency beyond our service level objectives. Management has issued a strict directive: we must reduce the model memory footprint by at least 50% (a 2x reduction in size) while ensuring that task evaluation degradation does not exceed 1.0% relative to the uncompressed baseline.

However, a naive uniform 8-bit or lower-bit quantization causes catastrophic accuracy collapse on our evaluation harness, failing our strict performance floor. The exact recipe—covering calibration data selection, outlier handling, layer sensitivity profiling, and mixed-precision allocation—remains an open engineering problem.

Your task is to build a rigorous, reproducible low-level quantization pipeline using pure NumPy. You must construct a reliable evaluation harness, build representative calibration sets, benchmark different quantization recipes under identical conditions, identify and protect sensitive layers via mixed precision, verify that both the size reduction and accuracy constraints are strictly met, and safeguard your workflow with robust regression tests that catch silent failures and metric regressions.
