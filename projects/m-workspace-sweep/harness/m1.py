import sys
import os
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        from sweep.engine import plan_engine
    except ImportError as e:
        return {"_note": f"Import failed: {e}"}

    ok = 0
    total = 0
    out = {}

    for i, (config, profile, dev_mem, limits) in enumerate(ref.SCENARIOS):
        for limit in limits:
            total += 1
            want_mem, want_lat = ref.plan_engine(config, profile, limit)
            try:
                got_mem, got_lat = plan_engine(config, profile, limit)
            except Exception as e:
                return {"_note": f"Exception on scenario {i}, limit {limit}: {e}"}

            mem_match = (want_mem == got_mem) or (abs(want_mem - got_mem) < 1e-5)
            lat_match = (want_lat == got_lat) or (abs(want_lat - got_lat) < 1e-5)

            if mem_match and lat_match:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"scenario {i} limit {limit}: got ({got_mem}, {got_lat}), want ({want_mem}, {want_lat})"

    out["matches"] = float(ok)
    out["total"] = float(total)
    return out
