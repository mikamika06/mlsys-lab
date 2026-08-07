import numpy as np
import ref


def check(workdir):
    out = {"mse_sweep_matched": 0.0, "monotonic_trend_valid": 0.0}
    try:
        from mlx_quant.sweep import sweep_group_size_mse
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {e}"
        return out

    weights = ref.generate_test_weights(shape=(256, 128), seed=2024)
    group_sizes = (32, 64, 128)

    try:
        got = sweep_group_size_mse(weights, group_sizes=group_sizes, bits=4)
        want = ref.sweep_group_size_mse(weights, group_sizes=group_sizes, bits=4)
    except Exception as e:
        out["_note"] = f"Execution error: {type(e).__name__}: {e}"
        return out

    if not isinstance(got, dict):
        out["_note"] = f"Expected dict return, got {type(got)}"
        return out

    matched = True
    for gs in group_sizes:
        if gs not in got:
            matched = False
            out["_note"] = f"Missing group size {gs} in returned dict"
            break
        if not np.isclose(got[gs], want[gs], rtol=1e-4, atol=1e-6):
            matched = False
            out["_note"] = f"Group size {gs} MSE mismatch: got {got[gs]}, want {want[gs]}"
            break

    if matched:
        out["mse_sweep_matched"] = 1.0

    if 32 in got and 64 in got and 128 in got:
        if got[32] <= got[64] + 1e-6 and got[64] <= got[128] + 1e-6:
            out["monotonic_trend_valid"] = 1.0
        else:
            out["_note"] = f"Non-monotonic trend observed across group sizes: {got}"

    return out
