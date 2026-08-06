import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    try:
        from exporter.budget import compute_kv_bytes, compute_max_context
    except Exception as e:
        return {
            "max_context_matched": 0.0,
            "bytes_matched": 0.0,
            "_note": f"Failed to import exporter.budget: {e}",
        }

    out = {"max_context_matched": 0.0, "bytes_matched": 0.0}
    max_matched = 0
    bytes_matched = 0
    total = len(ref.CONFIGS)

    for cfg in ref.CONFIGS:
        want_ctx = ref.compute_max_context(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["byte_budget"],
            cfg["dtype"],
        )
        got_ctx = compute_max_context(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["byte_budget"],
            cfg["dtype"],
        )

        want_bytes = ref.compute_kv_bytes(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            want_ctx,
            cfg["dtype"],
        )
        got_bytes = compute_kv_bytes(
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            got_ctx,
            cfg["dtype"],
        )

        if got_ctx == want_ctx:
            max_matched += 1
        if got_bytes == want_bytes:
            bytes_matched += 1

    out["max_context_matched"] = 1.0 if max_matched == total else 0.0
    out["bytes_matched"] = 1.0 if bytes_matched == total else 0.0
    return out
