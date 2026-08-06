# Linear Chain vs Branching Tree Speculative Verification

We are optimizing speculative decoding throughput for local LLM inference engines. Our current implementation uses a fixed linear speculative draft chain of length $K$. However, under high latency conditions or low acceptance probabilities, linear draft verification suffers from diminishing returns, leading to wasted target model forward passes.

We want to transition our verification harness to support tree-structured speculative draft verification (such as Tree Attention / SpecInfer style evaluation) and compare its expected accepted token length against traditional linear chains under identical model sampling probability distributions.

Currently, our speculative engine lacks a unified metric harness to compute and verify:
1. Exact expected accepted length for a linear speculative chain given per-position acceptance probabilities or target/draft probability vectors.
2. Exact expected accepted length for an arbitrary tree-structured speculative draft (nodes, parent pointers, and acceptance probabilities) using tree verification logic (taking the longest valid accepted path from root to leaf).
3. Empirical verification and comparative analysis showing under which branching structures and acceptance distributions tree speculation strictly outperforms linear speculation for a fixed draft budget.

Implement the tree verification engine, linear/tree expected length calculators, and comparative test suite to ensure our engine accurately selects draft structures that maximize expected accepted tokens per step.
