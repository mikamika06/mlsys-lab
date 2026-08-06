import sys
import os

def check(workdir):
    harness_dir = os.path.dirname(os.path.abspath(__file__))
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    import ref

    sys.path.insert(0, workdir)
    try:
        from memattn.planner import max_sequence_length, max_batch_size
    except (ImportError, NotImplementedError) as e:
        return {"rel_err": 1.0, "_note": f"Import/Implementation error: {e}"}

    total_err = 0.0
    count = 0

    for cfg in ref.CONFIGS:
        for budget in ref.BUDGETS:
            for mode in ["lse", "prob"]:
                want_seq = ref.ref_max_sequence_length(
                    cfg["batch_size"], cfg["num_heads"], cfg["head_dim"], budget, mode=mode, dtype_bytes=cfg["dtype_bytes"]
                )
                want_batch = ref.ref_max_batch_size(
                    cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], budget, mode=mode, dtype_bytes=cfg["dtype_bytes"]
                )

                try:
                    got_seq = max_sequence_length(
                        cfg["batch_size"], cfg["num_heads"], cfg["head_dim"], budget, mode=mode, dtype_bytes=cfg["dtype_bytes"]
                    )
                    got_batch = max_batch_size(
                        cfg["seq_len"], cfg["num_heads"], cfg["head_dim"], budget, mode=mode, dtype_bytes=cfg["dtype_bytes"]
                    )
                except Exception as e:
                    return {"rel_err": 1.0, "_note": f"Execution error: {e}"}

                err_seq = abs(got_seq - want_seq) / max(1.0, float(want_seq))
                err_batch = abs(got_batch - want_batch) / max(1.0, float(want_batch))
                total_err += err_seq + err_batch
                count += 2

    mean_err = total_err / max(1, count)
    return {"rel_err": mean_err}
