import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"ratio_match": 0.0}
    try:
        from adapter_merge.sizing import checkpoint_stats
        ok = 0
        for b_shapes, l_shapes, dt in ref.SIZE_FIXTURES:
            want = ref.checkpoint_stats(b_shapes, l_shapes, dt)
            got = checkpoint_stats(b_shapes, l_shapes, dt)
            if (isinstance(got, dict) and
                got.get("base_bytes") == want["base_bytes"] and
                got.get("lora_bytes") == want["lora_bytes"] and
                abs(got.get("ratio", -1.0) - want["ratio"]) < 1e-6):
                ok += 1
        out["ratio_match"] = float(ok) / len(ref.SIZE_FIXTURES)
    except Exception as e:
        out["_note"] = f"Failed: {type(e).__name__}: {str(e)}"
    finally:
        sys.path.pop(0)
    return out
