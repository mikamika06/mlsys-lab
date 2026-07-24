import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    # Prepare random tensors
    a = rng.standard_normal((5, 3))
    b = rng.standard_normal((5, 3))
    x = rng.standard_normal((4, 6))

    # Keep originals for comparison
    orig_a = a.copy()
    orig_b = b.copy()
    orig_x = x.copy()

    # ---- functional_add ----
    copy_a = a.copy()
    copy_b = b.copy()
    out_add = sol.functional_add(copy_a, copy_b)
    expected_add = orig_a + orig_b
    err_out_add = np.max(np.abs(out_add - expected_add))
    err_inp_add_a = np.max(np.abs(copy_a - orig_a))
    err_inp_add_b = np.max(np.abs(copy_b - orig_b))

    # ---- functional_relu ----
    copy_x = x.copy()
    out_relu = sol.functional_relu(copy_x)
    expected_relu = np.maximum(0, orig_x)
    err_out_relu = np.max(np.abs(out_relu - expected_relu))
    err_inp_relu_x = np.max(np.abs(copy_x - orig_x))

    # ---- functional_copy ----
    copy_a2 = a.copy()
    out_copy = sol.functional_copy(copy_a2)
    expected_copy = orig_a.copy()
    err_out_copy = np.max(np.abs(out_copy - expected_copy))
    err_inp_copy_a = np.max(np.abs(copy_a2 - orig_a))

    # Aggregate errors
    max_abs_err_output = max(err_out_add, err_out_relu, err_out_copy)
    max_abs_err_input  = max(err_inp_add_a, err_inp_add_b,
                             err_inp_relu_x, err_inp_copy_a)

    return {
        "max_abs_err_output": float(max_abs_err_output),
        "max_abs_err_input": float(max_abs_err_input)
    }
