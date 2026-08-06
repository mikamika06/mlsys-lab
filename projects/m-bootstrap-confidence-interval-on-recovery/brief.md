We are evaluating the accuracy retention of a small language model after quantization, but our current deployment pipeline reports a single point-estimate for the recovery percentage without accounting for variance across evaluation samples. Because evaluation datasets like lm-eval consist of numerous discrete tasks and subset samples, a naive percentage drop or recovery score can be misleading due to sample noise, making it difficult to determine whether a quantized model's performance drop is statistically significant or merely an artifact of random sampling in the evaluation harness.

Your task is to implement a robust evaluation reporting module that runs an integrated evaluation routine, computes bootstrap confidence intervals on the recovery percentage compared to a baseline model, and safeguards the statistical evaluation pipeline against degenerate sampling distributions. 

You need to implement three core milestones:
1. Provide a module that executes or simulates running `lm-eval` style evaluations and extracts task-level scores for both baseline and quantized models.
2. Implement a bootstrap resampling method that computes empirical confidence intervals (e.g., 95% CI) on the recovery percentage metric.
3. Write a regression test suite in `tests/test_regression.py` that verifies the correctness of your bootstrap interval computations under simulated variations and detects improper sampling logic or inverted bounds.

Ensure that all calculations use deterministic random seeds and operate purely in Python and NumPy to guarantee reproducible and fast automated grading.
