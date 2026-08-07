import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"head_match": 0.0, "bytes_match": 0.0}
    try:
        from fp8kv.error import find_breaking_head
        from fp8kv.capacity import calc_cache_bytes
    except ImportError as e:
        out["_note"] = f"ImportError: {e}"
        return out

    x = ref.generate_fixture()

    try:
        w_head = ref.find_breaking_head(x)
        g_head = find_breaking_head(x)
        if w_head == g_head:
            out["head_match"] = 1.0
    except Exception as e:
        out["_note_head"] = str(e)

    try:
        w_b1 = ref.calc_cache_bytes(2048, 32, 128, 80, "fp8_per_head")
        g_b1 = calc_cache_bytes(2048, 32, 128, 80, "fp8_per_head")
        w_b2 = ref.calc_cache_bytes(2048, 32, 128, 80, "fp8_per_tensor")
        g_b2 = calc_cache_bytes(2048, 32, 128, 80, "fp8_per_tensor")

        if w_b1 == g_b1 and w_b2 == g_b2:
            out["bytes_match"] = 1.0
    except Exception as e:
        out["_note_bytes"] = str(e)

    return out
