import ref
from finetune.scaling import compute_memory_ratios

def check(workdir):
    out = {"ratios_matched": 0.0}
    try:
        got = compute_memory_ratios(ref.PARAM_COUNTS)
        ref_res = ref.compute_memory_ratios(ref.PARAM_COUNTS) if hasattr(ref, "compute_memory_ratios") else None

        # compute oracle directly
        fl_ref = []
        lq_ref = []
        for pc in ref.PARAM_COUNTS:
            mf = pc * 2 + pc * 2 + pc * 16 + pc * 0.2
            ml = pc * 2 + (pc * 0.01) * 2 + (pc * 0.01) * 16 + pc * 0.1
            mq = pc * 0.5 + (pc * 0.01) * 2 + (pc * 0.01) * 16 + pc * 0.05
            fl_ref.append(mf / ml)
            lq_ref.append(ml / mq)

        if "full_lora" not in got or "lora_qlora" not in got:
            out["_note"] = "Missing keys in scaling ratios dictionary"
            return out

        ok = True
        for a, b in zip(got["full_lora"], fl_ref):
            if abs(a - b) > 1e-5:
                ok = False
        for a, b in zip(got["lora_qlora"], lq_ref):
            if abs(a - b) > 1e-5:
                ok = False

        out["ratios_matched"] = 1.0 if ok else 0.0
    except Exception as e:
        out["_note"] = f"Exception in milestone 2: {e}"
        out["ratios_matched"] = 0.0
    return out
