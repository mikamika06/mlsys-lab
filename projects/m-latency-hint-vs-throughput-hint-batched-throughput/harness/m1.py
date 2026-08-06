import os
import tempfile
import ref


def check(workdir):
    out = {"configs_matched": 0.0, "cache_speedup": 0.0}
    from cpuhints.scheduler import derive_config, compile_model

    ok = True
    for cores in ref.CORES_TO_TEST:
        for hint in ["latency", "throughput"]:
            try:
                if derive_config(hint, cores) != ref.derive_config(hint, cores):
                    ok = False
            except Exception:
                ok = False

    out["configs_matched"] = 1.0 if ok else 0.0

    try:
        with tempfile.TemporaryDirectory() as d:
            t1 = compile_model("test_model", d)
            t2 = compile_model("test_model", d)
            if t1 > 0 and t2 > 0:
                out["cache_speedup"] = t1 / t2
    except Exception:
        pass

    return out
