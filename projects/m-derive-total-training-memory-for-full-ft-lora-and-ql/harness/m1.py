import ref
from finetune.memory import compute_training_memory

def check(workdir):
    out = {"memory_estimates_matched": 0.0}
    ok = True
    for pc in ref.PARAM_COUNTS:
        for m in ref.METHODS:
            want = ref.compute_training_memory(pc, m) if hasattr(ref, "compute_training_memory") else 0
            # compute using reference logic directly as oracle
            if hasattr(ref, "compute_training_memory"):
                pass
            # standard reference implementation comparison
            if m == "full":
                w = pc * 2 + pc * 2 + pc * 16 + pc * 0.2
            elif m == "lora":
                w = pc * 2 + (pc * 0.01) * 2 + (pc * 0.01) * 16 + pc * 0.1
            else:
                w = pc * 0.5 + (pc * 0.01) * 2 + (pc * 0.01) * 16 + pc * 0.05

            try:
                got = compute_training_memory(pc, m)
                if abs(got - w) > 1e-5:
                    ok = False
                    out["_note"] = f"Mismatch for param_count={pc}, method={m}: got {got}, want {w}"
            except Exception as e:
                ok = False
                out["_note"] = f"Exception raised for param_count={pc}, method={m}: {e}"
                break
    out["memory_estimates_matched"] = 1.0 if ok else 0.0
    return out
