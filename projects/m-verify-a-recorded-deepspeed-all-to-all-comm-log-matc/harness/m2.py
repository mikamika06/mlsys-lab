import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from sp_comm.mem_budget import max_seq_len
    except ImportError:
        sys.path.pop(0)
        return {"matches": 0, "_note": "failed to import max_seq_len"}

    matches = 0
    note = ""
    for cfg in ref.MEM_BUDGET_FIXTURES:
        try:
            want = ref.max_seq_len(**cfg)
            got = max_seq_len(**cfg)
            if want == got:
                matches += 1
            else:
                note = f"mismatch: want {want}, got {got}"
        except Exception as e:
            sys.path.pop(0)
            return {"matches": matches, "_note": f"crash: {e}"}

    out = {"matches": matches}
    if matches < len(ref.MEM_BUDGET_FIXTURES) and note:
        out["_note"] = note

    sys.path.pop(0)
    return out
