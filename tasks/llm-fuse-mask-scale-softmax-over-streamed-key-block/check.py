import numpy as np

def _oracle(keys, values, mask, scale):
    # Compute the scaled dot-product attention with masking and softmax
    scores = np.dot(keys, values.T) * scale
    scores += mask  # Apply the mask
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # Stable softmax
    softmax_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return softmax_scores

def grade(sol, fx) -> dict:
    # Example inputs
    keys_np = np.random.rand(10, 64)  # 10 keys of dimension 64
    values_np = np.random.rand(10, 64)  # 10 values of dimension 64
    mask_np = np.random.rand(10, 10) * -1e9  # Causal mask
    scale = 1 / np.sqrt(64)  # Scaling factor

    keys = keys_np.tolist()
    values = values_np.tolist()
    mask = mask_np.tolist()

    # Get the output from the student's solution
    try:
        got = sol.fuse_mask_scale_softmax(keys, values, mask, scale)
    except Exception:
        return {"max_abs_err": 1.0}  # Return a high error if there's an exception

    # Compute the reference output
    expected = _oracle(keys_np, values_np, mask_np, scale)

    # Calculate the maximum absolute error
    error = np.max(np.abs(np.array(got) - expected))
    return {"max_abs_err": error}
