# High Rejection Rate in Token-Only Speculative Decoding Draft Heads

Our speculative execution pipeline using a lightweight token-only draft head on `distilgpt2` features is experiencing high candidate rejection rates during vLLM verification runs. Token-only draft heads predict candidate draft tokens based exclusively on previously emitted token IDs. Consequently, they miss crucial contextual feature representations from the target LLM's top-layer hidden states, leading to poor draft acceptance lengths and sub-optimal speculative speedups.

To increase candidate acceptance without introducing excessive parameter overhead, we must replace the token-only draft head with an EAGLE-style feature-conditioned draft head. The EAGLE architecture conditions draft prediction on a fusion of token embeddings and target model hidden-state features.

Your task is to build and evaluate this feature-conditioned draft system and verify its efficiency metrics:
1. Implement a lightweight EAGLE-style feature-conditioned draft head that combines token embeddings with target hidden features to predict next-token logits.
2. Build an evaluation utility comparing token-only draft prediction accuracy against hidden-state feature-conditioned draft prediction accuracy.
3. Construct a parser to reconstruct the mean accepted token length and total verification steps from recorded vLLM EAGLE-3 execution logs.
4. Implement regression tests in `tests/test_regression.py` validating feature-conditioned accuracy advantages and verifying log analysis correctness under fault injection.
