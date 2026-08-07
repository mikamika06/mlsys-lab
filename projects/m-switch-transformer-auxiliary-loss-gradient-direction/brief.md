# Symptom Report: Unstable Expert Routing and Load Balancing Failures in MoE Training

During distributed training of our Mixture-of-Experts (MoE) models, we are seeing severe load imbalance across experts, leading to straggler steps and out-of-memory errors on over-allocated workers.

We attempted to stabilize expert assignment by applying a standard Switch Transformer auxiliary load-balancing loss ($L_{\text{aux}} = \alpha \cdot N \sum P_i f_i$). However, analysis of the router weight gradients reveals that the auxiliary loss gradient direction is pushing router weights in unexpected directions relative to the true expert assignment probabilities, sometimes exacerbating routing collapse rather than preventing it.

In parallel, recent work on aux-loss-free load balancing (such as DeepSeek-V3's dynamic expert bias updates) promises to maintain expert balance without corrupting the primary routing gradients. We need to evaluate whether bias-update mechanics achieve faster convergence toward balanced expert allocation compared to loss-based auxiliary objectives without sacrificing routing accuracy.

## Required Work

1. Implement the Switch Transformer auxiliary loss computation and compute the exact analytical gradient of $L_{\text{aux}}$ with respect to router logits, verifying the exact gradient direction vector.
2. Implement a step-by-step simulator for DeepSeek-V3 dynamic expert bias updates and track expert load balance convergence over training iterations.
3. Compare the convergence rate and final coefficient of variation ($\text{CV}$) of expert loads between standard Switch auxiliary loss and aux-loss-free bias tracking across synthetic routing sequences.
4. Add regression tests in `tests/test_regression.py` validating that router updates maintain negative feedback loops on over-subscribed experts and preserve routing invariants under gradient perturbations.
