# Serving Preflight Validator for Quantized vLLM Deployments

Our production serving clusters frequently experience out-of-memory (OOM) crashes and silent throughput degradation when launching vLLM instances with quantized model weights. Deployment scripts currently push configurations to GPU nodes without verifying whether the target quantization format fits on the available GPU memory topology, nor whether the resulting token throughput actually improves upon standard FP16 execution.

You are tasked with building a preflight validation module (`preflight/`) that evaluates deployment candidates prior to launch. The module must estimate peak GPU memory requirements across multi-GPU setups for given quantization configurations (AWQ, GPTQ, FP8, Unquantized FP16) and calculate projected token throughput ratios against an FP16 baseline.

Finally, you must write a suite of regression tests in `tests/test_regression.py` that catches invalid preflight deployment approvals and ensures memory overflow conditions are properly blocked.
