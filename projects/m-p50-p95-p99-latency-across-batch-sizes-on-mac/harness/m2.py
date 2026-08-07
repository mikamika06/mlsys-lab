import ref
import numpy as np


def check(workdir):
    try:
        from latbench.tradeoff import derive_optimal_batch_sizes
        from latbench.shapes import compare_shape_throughput
    except ImportError as e:
        return {"optimal_matched": 0.0, "shapes_matched": 0.0, "_note": f"Import error: {e}"}

    data = ref.generate_synthetic_data(seed=456)
    p_summary = ref.analyze_batch_latencies(data["profile_data"])
    slo = data["slo"]

    want_opt = ref.derive_optimal_batch_sizes(p_summary, slo)
    try:
        got_opt = derive_optimal_batch_sizes(p_summary, slo)
    except Exception as e:
        return {"optimal_matched": 0.0, "shapes_matched": 0.0, "_note": f"Execution error in tradeoff: {e}"}

    opt_matched = (got_opt == want_opt)

    static_lats = data["static_lats"]
    dynamic_lats = data["dynamic_lats"]
    batch_size = data["shape_batch"]

    want_shapes = ref.compare_shape_throughput(static_lats, dynamic_lats, batch_size)
    try:
        got_shapes = compare_shape_throughput(static_lats, dynamic_lats, batch_size)
    except Exception as e:
        return {"optimal_matched": 1.0 if opt_matched else 0.0, "shapes_matched": 0.0, "_note": f"Execution error in shapes: {e}"}

    shapes_matched = True
    if not isinstance(got_shapes, dict):
        shapes_matched = False
    else:
        for k, v in want_shapes.items():
            if k not in got_shapes or not np.isclose(got_shapes[k], v, rtol=1e-3):
                shapes_matched = False
                break

    return {
        "optimal_matched": 1.0 if opt_matched else 0.0,
        "shapes_matched": 1.0 if shapes_matched else 0.0,
        "_note": "Milestone 2 checks passed" if (opt_matched and shapes_matched) else "Mismatch in optimization or shape metrics"
    }
