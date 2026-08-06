import numpy as np
import ref


def check(workdir):
    out = {"max_abs_err": 100.0, "axes_mapped": 0.0}
    try:
        from tirsched.schedule import create_naive_matmul, apply_split_reorder_vectorize_parallel, execute_tir_matmul
        from tirsched.analysis import print_tir_loop_nest, map_axes_to_transforms
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    m, n, k, a_np, b_np, c_ref = ref.generate_fixtures(seed=2024)
    naive_mod = create_naive_matmul(m, n, k)

    try:
        steps = apply_split_reorder_vectorize_parallel(naive_mod, factors=(16, 16))
    except Exception as e:
        out["_note"] = f"Failed to apply schedule steps: {e}"
        return out

    if not isinstance(steps, list) or len(steps) != 4:
        out["_note"] = f"Expected 4 schedule steps, got {len(steps) if isinstance(steps, list) else type(steps)}"
        return out

    max_err = 0.0
    for name, mod in steps:
        try:
            out_c = execute_tir_matmul(mod, a_np, b_np)
            err = float(np.max(np.abs(out_c - c_ref)))
            if err > max_err:
                max_err = err
        except Exception as e:
            out["_note"] = f"Execution failed at step {name}: {e}"
            return out

    out["max_abs_err"] = max_err

    final_mod = steps[-1][1]
    nest_str = print_tir_loop_nest(final_mod)
    mapping = map_axes_to_transforms(nest_str)

    expected_axes = {"i_outer", "j_outer", "i_inner", "k", "j_inner"}
    if isinstance(mapping, dict) and expected_axes.issubset(set(mapping.keys())):
        if mapping.get("j_inner") == "vectorize" and mapping.get("i_outer") == "parallel":
            out["axes_mapped"] = 1.0
        else:
            out["_note"] = f"Incorrect axis mappings: {mapping}"
    else:
        out["_note"] = f"Missing mapped axes. Got keys: {list(mapping.keys()) if isinstance(mapping, dict) else mapping}"

    return out
