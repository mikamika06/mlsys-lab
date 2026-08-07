import ref

def check(workdir):
    from runner.memory import ModelMemoryProfiler

    m = {"formula_ok": 0.0}
    prof = ModelMemoryProfiler(4000, 32, 4096, 32)
    val = prof.kv_cache_size_mb(2048, 2)
    expected = ref.get_oracle_kv(2048, 2, 32, 4096)
    if abs(val - expected) < 1e-5:
        m["formula_ok"] = 1.0
    return m
