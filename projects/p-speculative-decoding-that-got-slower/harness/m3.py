import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        import specdec.analyzer as an
    except ImportError:
        return {"table_len_ok": 0.0, "table_vals_ok": 0.0}

    m = {"table_len_ok": 0.0, "table_vals_ok": 0.0}
    try:
        table = an.batch_speedup_table(0.8, 4, 16, ref.cost_model)
        if len(table) == 16:
            m["table_len_ok"] = 1.0
            v1 = an.compute_speedup(0.8, 4, *ref.cost_model(1, 4))
            v8 = an.compute_speedup(0.8, 4, *ref.cost_model(8, 4))
            if abs(table[0] - v1) < 1e-5 and abs(table[7] - v8) < 1e-5:
                m["table_vals_ok"] = 1.0
    except Exception:
        pass
    return m
