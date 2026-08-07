import ref
import os
import tempfile

def check(workdir):
    from qcache.shape import static_cache_shapes
    from qcache.persist import save_cache, load_cache

    shape_ok = 1.0
    for cfg in ref.CONFIGS:
        got = static_cache_shapes(cfg)
        want = ref.ref_static_cache_shapes(cfg)
        if got != want:
            shape_ok = 0.0
            break

    persist_ok = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.pkl")
        dummy = [1, 2, 3]
        try:
            save_cache(dummy, path)
            loaded = load_cache(path)
            if loaded == dummy:
                persist_ok = 1.0
        except Exception:
            pass

    return {"shape_match": shape_ok, "persist_match": persist_ok}
