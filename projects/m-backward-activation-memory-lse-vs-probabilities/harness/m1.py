import sys
import os

def check(workdir):
    harness_dir = os.path.dirname(os.path.abspath(__file__))
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    import ref

    sys.path.insert(0, workdir)
    try:
        from memattn.allocator import compute_activation_memory
    except (ImportError, NotImplementedError) as e:
        return {"rel_err": 1.0, "_note": f"Import/Implementation error: {e}"}

    total_err = 0.0
    count = 0

    for cfg in ref.CONFIGS:
        for mode in ["lse", "prob"]:
            want = ref.ref_compute_activation_memory(
                cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], mode=mode, dtype_bytes=cfg["dtype_bytes"]
            )
            try:
                got = compute_activation_memory(
                    cfg["batch_size"], cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], mode=mode, dtype_bytes=cfg["dtype_bytes"]
                )
            except Exception as e:
                return {"rel_err": 1.0, "_note": f"Execution error: {e}"}

            err = abs(got - want) / max(1.0, float(want))
            total_err += err
            count += 1

    mean_err = total_err / max(1, count)
    return {"rel_err": mean_err}
