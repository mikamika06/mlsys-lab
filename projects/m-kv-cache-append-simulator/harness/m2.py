import os
import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from simulator.core import decode_bandwidth

    out = {"exact_match": 0.0, "total": float(len(ref.FIXTURES))}
    ok = 0
    for i, fx in enumerate(ref.FIXTURES):
        try:
            want = ref.decode_bandwidth(
                fx["cache_seqlens"], fx["num_layers"], fx["num_kv_heads"],
                fx["head_dim"], fx["dtype_bytes"]
            )
            got = decode_bandwidth(
                fx["cache_seqlens"], fx["num_layers"], fx["num_kv_heads"],
                fx["head_dim"], fx["dtype_bytes"]
            )
            if want == got:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"fixture {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"crash on fixture {i}: {e}"
    if ok == len(ref.FIXTURES):
        out["exact_match"] = 1.0
    return out
