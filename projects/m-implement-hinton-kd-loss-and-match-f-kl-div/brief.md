An internal audit of our model distillation pipeline has flagged several subtle correctness and performance issues in our custom implementation of Hinton Knowledge Distillation (KD). Specifically, training runs using our `kd_loss` module diverge or fail to converge to the expected teacher-student parity, exhibiting mysterious gradient scaling anomalies and incorrect probability distribution scaling.

When engineers attempted to implement standard temperature-scaled Softmax and Cross-Entropy distillation following Hinton's seminal recipe, several symptoms emerged:
1. The loss values and gradients do not match native PyTorch functional utilities like `F.kl_div` when properly configured for log-probabilities versus standard probabilities, leading to mismatched scaling factors of $T^2$.
2. The trade-off between the soft distillation loss and the hard label loss behaves erratically because the temperature scaling factor is incorrectly applied to the gradients or omitted from the backward pass normalization.
3. Diagnostic scripts checking the softmax entropy across varying temperature parameters $T$ reveal incorrect sharpness curves, failing to capture the expected smoothing effect on logits before computing KL divergence.
4. Without a robust regression test suite, subtle changes to the loss formulation re-introduce the missing $T^2$ gradient scaling bug, causing student models trained with high temperatures to suffer from suppressed soft-target gradients.

Your task is to fix our distillation library. You need to correctly implement Hinton KD loss matching `F.kl_div` behavior with proper $T^2$ scaling, compute temperature-vs-softmax-entropy curves correctly, and write a rigorous regression test that catches implementations missing the mandatory $T^2$ scaling factor.
