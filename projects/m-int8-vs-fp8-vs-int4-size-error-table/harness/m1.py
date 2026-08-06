import ref
import numpy as np

def check(workdir):
    from qsim.eval import build_table

    rs = np.random.RandomState(42)
    weights = rs.randn(128, 256).astype(np.float32) * 2.0

    out = {"table_matches": 0.0}
    want = ref.build_table(weights)

    try:
        got = build_table(weights)
    except Exception as e:
        out["_note"] = f"Error during build_table: {e}"
        return out

    for k in want:
        if k not in got:
            out["_note"] = f"Missing key '{k}' in returned table"
            return out
        if not np.isclose(want[k]["size_ratio"], got[k]["size_ratio"]):
            out["_note"] = f"size_ratio mismatch for {k}: expected {want[k]['size_ratio']}, got {got[k]['size_ratio']}"
            return out
        if not np.isclose(want[k]["mse"], got[k]["mse"]):
            out["_note"] = f"mse mismatch for {k}: expected {want[k]['mse']}, got {got[k]['mse']}"
            return out

    out["table_matches"] = 1.0
    return out
