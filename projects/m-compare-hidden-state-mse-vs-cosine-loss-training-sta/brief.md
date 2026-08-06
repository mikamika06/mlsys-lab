# Compare Hidden State MSE vs Cosine Loss Training Stability

When distilling knowledge from large models, aligning the hidden states of a student network to a teacher network is a highly effective technique. However, the choice of distance metric—typically Mean Squared Error (MSE) or Cosine Similarity (Cosine Loss)—drastically impacts gradient scale and overall training stability.

In this project, you will implement both loss functions and compare their training stability by tracking gradient norms during a simulated hidden-state alignment task.

## Milestones

1. **Loss Implementations:** Implement both `mse_hidden_loss` and `cosine_hidden_loss` functions in PyTorch to compute the distance between teacher and student representations.
2. **Gradient Tracking:** Implement the `track_gradient_norms` utility within the training step to capture the L2 norm of the student's gradients after the backward pass.
3. **Stability Comparison:** Run a training loop for both objectives, compute the variance of the tracked gradient norms, and assert that the cosine loss provides a lower gradient variance (higher stability) compared to raw un-normalized MSE.
