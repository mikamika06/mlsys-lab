import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from kvcapacity.feasibility import concurrency_ceiling, build_feasibility_matrix

    out = {"concurrency_matched": 0.0, "matrix_matched": 0.0}

    conc_ok = 0
    total_conc_tests = 0
    for cfg in ref.CONFIGS:
        for gpu in ref.GPU_CONFIGS:
            for tp in ref.TP_OPTIONS:
                for m_dt in ref.MODEL_DTYPES:
                    for k_dt in ref.KV_DTYPES:
                        for seq in [32768, 131072]:
                            total_conc_tests += 1
                            want = ref.concurrency_ceiling(gpu["memory_gb"], cfg, tp, m_dt, k_dt, seq)
                            got = concurrency_ceiling(gpu["memory_gb"], cfg, tp, m_dt, k_dt, seq)
                            if got == want:
                                conc_ok += 1
                            elif "_note" not in out:
                                out["_note"] = f"concurrency mismatch gpu={gpu['name']} tp={tp} seq={seq}: got {got}, want {want}"

    if conc_ok == total_conc_tests:
        out["concurrency_matched"] = 1.0

    matrix_ok = 0
    total_matrix_tests = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        want_mat = ref.build_feasibility_matrix(cfg, 131072, ref.GPU_CONFIGS, ref.TP_OPTIONS, ref.MODEL_DTYPES, ref.KV_DTYPES)
        got_mat = build_feasibility_matrix(cfg, 131072, ref.GPU_CONFIGS, ref.TP_OPTIONS, ref.MODEL_DTYPES, ref.KV_DTYPES)
        if got_mat == want_mat:
            matrix_ok += 1
        elif "_note" not in out:
            out["_note"] = f"matrix mismatch for config {cfg.get('name')}"

    if matrix_ok == total_matrix_tests:
        out["matrix_matched"] = 1.0

    return out
