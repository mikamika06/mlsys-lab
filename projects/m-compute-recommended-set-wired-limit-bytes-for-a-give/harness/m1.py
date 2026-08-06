import ref
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from limits.compute import recommend_wired_limit
    except ImportError:
        return {"configs_matched": 0.0, "_note": "limits/compute.py missing or unimportable"}

    gb = 1024 ** 3
    memsizes = [8, 16, 24, 32, 64, 96, 128, 192]

    ok = 0
    out = {"configs_matched": 0.0, "total": float(len(memsizes))}

    for m in memsizes:
        mem_bytes = m * gb
        want = ref.recommend_wired_limit(mem_bytes)
        try:
            got = recommend_wired_limit(mem_bytes)
        except NotImplementedError:
            out["_note"] = "recommend_wired_limit not implemented"
            break

        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"For {m}GB: got {got}, expected {want}"

    out["configs_matched"] = float(ok)
    return out
