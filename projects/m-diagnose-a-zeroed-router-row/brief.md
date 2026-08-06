# Ticket: Router output anomaly and parameter count mismatch on MoE rollout

## Context
During the evaluation of our top-k Mixture-of-Experts (MoE) routing stack for long-context workloads, two distinct issues were flagged by the core ML infrastructure team.

First, monitoring on routed expert activations showed anomalous behavior during certain token routing evaluations: specific sequence positions produce router probability vectors where a row is completely zeroed out after top-k selection and normalization. This causes downstream numerical instability and invalid expert dispatch schedules in the MoE layer.

Second, the capacity planning calculator reported inconsistent values for active versus total parameter counts when evaluating candidate MoE layer configurations, leading to errors in memory provisioning and flop estimation.

## Task
You need to inspect the MoE routing and parameter accounting code in `moerouter/` and resolve both issues across three milestones:

1. **Active vs Total Parameter Accounting:** Implement configuration parsing in `moerouter/params.py` to calculate total parameters (including non-expert base parameters and all expert weights) and active parameters per token (base parameters plus activated top-k expert parameters per MoE layer) for an MoE model configuration.
2. **Diagnose and Fix Zeroed Router Rows:** Implement top-k gating with numerical safeguards in `moerouter/routing.py`. Identify the edge cases where router logits or masking produce zeroed-out probability rows across all selected experts (e.g., severe negative logit masking or all-masked expert candidates), diagnose the underlying condition, and safeguard the normalization step to guarantee valid non-zero probability distributions over selected experts.
3. **Safeguard Integration via Regression Testing:** Provide regression tests in `tests/test_regression.py` that enforce top-k router output sanity and parameter accounting invariants. Your tests will be verified by introducing a simulated router degradation fault (where top-k router normalization yields zeroed rows on extreme masked inputs), which your test suite must catch.
