import numpy as np
import ref


def check(workdir):
    from fused_grad.backward import (
        finite_difference_grad_x,
        fused_elementwise_backward,
        fused_elementwise_forward,
    )

    out = {"backward_matches_fd": 0.0}
    cases = ref.generate_test_cases()
    matched = 0

    for i, c in enumerate(cases):
        fwd = fused_elementwise_forward(c["x"], c["index_map"])
        if len(fwd) != len(c["index_map"]):
            out["_note"] = f"Case {i}: output length mismatch"
            return out

        bwd = fused_elementwise_backward(c["grad_output"], c["x"], c["index_map"], use_atomic=True)
        fd = finite_difference_grad_x(c["x"], c["index_map"], c["grad_output"], eps=1e-5)

        if np.allclose(bwd, fd, rtol=1e-3, atol=1e-3):
            matched += 1
        else:
            out["_note"] = f"Case {i}: bwd and fd do not match. Max diff: {np.max(np.abs(bwd - fd)):.6f}"
            return out

    if matched == len(cases):
        out["backward_matches_fd"] = 1.0

    return out
