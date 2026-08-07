We noticed surprising downstream model routing behavior when evaluating our Mixture-of-Experts (MoE) block implementations. Small numerical discrepancies in router logits caused noticeable shifts in expert token distribution depending on how routing gating was computed. Furthermore, tensor dispatch masks produced by custom kernel paths did not align with reference dispatch structures.

To stabilize our routing pipeline, we need to carefully benchmark and implement low-level gating functions and dispatch representations.

Your task:
1. Implement both `top_k_then_softmax` and `softmax_then_top_k` routing gating. Compute maximum absolute numerical divergence and cosine similarity between the resulting gating weights on identical router logits.
2. Build an exact reconstruction of `MixtralSparseMoeBlock`'s token dispatch tensor `(num_experts, max_tokens, top_k)` that routes tokens to expert indices based on selected gating assignments.
3. Compute router-logit Shannon entropy per layer and per token across a multi-layer MoE model, identifying layers where routing choices are overly concentrated or nearly uniform.
4. Implement a regression test suite in `tests/test_regression.py` that validates gating invariants and catches invalid dispatch indexing.
