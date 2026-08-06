import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from aotbreak.profiler import parse_overhead_records

    records = ref.generate_records()
    want = ref.parse_overhead_records(records)
    got = parse_overhead_records(records)

    out = {"profiles_matched": 0.0, "rel_err": 1.0}

    if not isinstance(got, dict) or set(got.keys()) != set(want.keys()):
        out["_note"] = f"Workload keys mismatch: got {list(got.keys()) if isinstance(got, dict) else type(got)}, expected {list(want.keys())}"
        return out

    max_rel_err = 0.0
    matched = True
    for w in want:
        for k in ["jit_compile_ms", "jit_exec_ms", "aot_load_ms", "aot_exec_ms"]:
            w_val = want[w][k]
            g_val = got[w].get(k, 0.0)
            err = abs(g_val - w_val) / max(1e-9, abs(w_val))
            if err > max_rel_err:
                max_rel_err = err
            if err > 1e-5:
                matched = False

    out["rel_err"] = float(max_rel_err)
    out["profiles_matched"] = 1.0 if matched else 0.0
    return out
