# Acceptance rate fundamentals in speculative decoding

In speculative decoding, the efficiency of speculative drafting depends heavily on how the acceptance rate behaves under different conditions. In our production monitoring pipelines, we noticed that using a single scalar acceptance rate across all draft positions misleads resource allocation and draft length selection strategies.

To fix this, we need to move beyond flat metrics and build fine-grained analysis tools:

1. **Positional Decay Analysis**: Given raw speculative trace logs containing binary acceptance decisions across generated tokens, calculate both the aggregate flat acceptance rate and the per-position acceptance decay curve.
2. **Workload Variation Assessment**: Compute the expected overall acceptance rate for different prompt domains (such as code, chat, and summarization) given position-dependent acceptance probabilities, and observe how domain differences shift draft effectiveness.
3. **Theoretical Bounds**: Derive the expected token acceptance probability based on the Kullback-Leibler (KL) divergence $KL(P_{\text{draft}} \parallel P_{\text{target}})$ using the theoretical lower bound $1 - \frac{1}{2} \|P_{\text{draft}} - P_{\text{target}}\|_1$ via Pinsker's inequality and total variation bounds.
4. **Regression Safeguard**: Implement unit tests in `tests/test_regression.py` that verify these metrics and explicitly detect when downstream code erroneously flat-averages position probabilities instead of correctly tracking per-position decay curves.
