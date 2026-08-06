import os
import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from simulator.core import append_tokens

    out = {"matches": 0.0, "total": float(len(ref.FIXTURES))}
    ok = 0
    for i, fx in enumerate(ref.FIXTURES):
        try:
            want = ref.append_tokens(fx["cache_seqlens"], fx["block_tables"], fx["block_size"])
            got = append_tokens(fx["cache_seqlens"], fx["block_tables"], fx["block_size"])
            if want == got:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"fixture {i}: got {got[:2]}, want {want[:2]}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"crash on fixture {i}: {e}"
    out["matches"] = float(ok)
    return out
