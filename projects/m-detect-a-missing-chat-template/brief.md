We are experiencing critical failures in our post-training quantization calibration pipeline when preparing models for low-precision deployment under the `rw2-quant-libs` track. During the recent integration runs, a subset of incoming model configurations and dataset streams caused severe silent corruptions or downstream perplexity regressions during calibration, without throwing immediate pipeline crashes.

Specifically, the calibration routine occasionally attempts to process raw token inputs directly when a model's chat template or tokenizer chat formatting specification is entirely absent from the model repository or configuration artifact, leading to malformed prompt tokens and catastrophic out-of-distribution activation statistics. Furthermore, when computing Hessian matrices for second-order quantization methods, our current memory footprint balloons uncontrollably because we fail to appropriately optimize and budget the sequence length and batch sizing parameters, resulting in out-of-memory errors on target hardware.

Your task is to build a robust diagnostic and configuration suite under `calib/` that addresses these issues systematically:
1. Implement a precise detector that identifies missing or unformatted chat templates before calibration data generation begins, raising an explicit diagnostic error or fallback structure.
2. Implement an optimization utility that selects the optimal batch size (N) and sequence length (seqlen) parameters to minimize calibration compute and memory overhead while respecting hardware constraints.
3. Track, calculate, and bound Hessian accumulation memory requirements dynamically during the calibration process to prevent memory leaks and out-of-memory faults.

You must deliver the implementation files under `calib/`, ensure the correct files are structured, and supply a comprehensive regression test suite in `tests/test_regression.py` that successfully catches faulty Hessian memory accounting.
