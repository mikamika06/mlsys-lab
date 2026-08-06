import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.ref as ref


def check(workdir):
    from guardeval.evaluator import evaluate_graph_guards, evaluate_guard

    out = {"evaluations_correct": 0.0}

    g_shape = {"type": "shape", "dim": 0, "val": 16}
    g_dtype = {"type": "dtype", "val": "float32"}
    g_stride = {"type": "stride", "dim": 1, "val": 1}
    g_contig = {"type": "contiguous", "val": True}

    m_pass = {"shape": (16, 32), "dtype": "float32", "strides": (32, 1), "is_contiguous": True}
    m_fail = {"shape": (8, 32), "dtype": "float32", "strides": (32, 1), "is_contiguous": True}

    ok1, r1 = evaluate_guard(g_shape, m_pass)
    ok2, r2 = evaluate_guard(g_shape, m_fail)

    if not ok1 or r1 is not None:
        out["_note"] = "Valid guard evaluation failed"
        return out

    if ok2 or r2 != "shape[0] == 16":
        out["_note"] = f"Expected shape failure reason 'shape[0] == 16', got '{r2}'"
        return out

    guards = [g_shape, g_dtype, g_stride, g_contig]
    ok_g, _ = evaluate_graph_guards(guards, m_pass)
    ok_f, r_f = evaluate_graph_guards(guards, m_fail)

    if ok_g and (not ok_f) and r_f == "shape[0] == 16":
        out["evaluations_correct"] = 1.0
    else:
        out["_note"] = f"Graph guard evaluation mismatch: ok_g={ok_g}, ok_f={ok_f}, r_f={r_f}"

    return out
