import sys
sys.path.insert(0, ".")
import ref

def check(workdir):
    out = {"is_legacy_correct": 0.0, "port_correct": 0.0}
    try:
        from kvcache.legacy import is_legacy_tuple, port_legacy_to_cache
        valid = ref.generate_legacy_tuple()
        invalid1 = {"layer0": valid[0]}
        invalid2 = (valid[0][0], valid[0][1])

        ok1 = is_legacy_tuple(valid) is True
        ok2 = is_legacy_tuple(invalid1) is False
        ok3 = is_legacy_tuple(invalid2) is False
        if ok1 and ok2 and ok3:
            out["is_legacy_correct"] = 1.0

        cache = port_legacy_to_cache(valid)
        if cache.get_seq_length(0) == valid[0][0].shape[-2]:
            out["port_correct"] = 1.0
    except Exception as e:
        out["_note"] = f"m1 failed with {type(e).__name__}: {str(e)}"
    return out
