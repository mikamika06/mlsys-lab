We have a legacy NumPy implementation of a Radial Basis Function (RBF) interaction layer that computes pairwise distance interactions between two sets of features. We need to wrap it as a PyTorch custom operation so that it seamlessly participates in compiled models (`torch.compile`) and autograd loops.

A previous intern tried to add a fake kernel (`register_fake`) to make `torch.compile` happy, but they got the output dimension wrong: their fake kernel assumed the output shape was `(B, N, D)` instead of the correct `(B, N, M)`. This caused PyTorch's shape propagation to crash during compilation.

Your tasks:
1. Implement the NumPy forward and VJP (vector-Jacobian product) functions for the RBF layer in `custom_op/rbf.py`.
2. Wrap the forward function as a custom op `mylib::rbf` using `@torch.library.custom_op`.
3. Write a correct fake kernel for it using `@rbf_interact.register_fake`, ensuring the output shape is `(B, N, M)`.
4. Wire up autograd using `rbf_interact.register_autograd(backward, setup_context=setup_context)` so that `backward()` routes gradients through your NumPy VJP.
5. Provide a safety net in `tests/test_regression.py` that uses `torch.autograd.gradcheck` to catch any gradient regressions if the underlying NumPy logic changes in the future.
