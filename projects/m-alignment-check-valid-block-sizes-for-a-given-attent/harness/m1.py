import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref

    try:
        from blockalign.validator import filter_valid_block_sizes, validate_block_size
    except Exception as e:
        return {
            "validations_matched": 0.0,
            "filters_matched": 0.0,
            "_note": f"Failed to import validator module: {type(e).__name__}: {e}",
        }

    out = {
        "validations_matched": 0.0,
        "filters_matched": 0.0,
        "total_configs": float(len(ref.CONFIGS_M1)),
    }

    val_ok = 0
    filter_ok = 0

    for i, cfg in enumerate(ref.CONFIGS_M1):
        backend = cfg["backend"]
        model = cfg["model"]
        candidates = cfg["candidates"]

        ref_val_results = [ref.validate_block_size(backend, model, bs) for bs in candidates]
        try:
            got_val_results = [validate_block_size(backend, model, bs) for bs in candidates]
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"M1 config {i} validate_block_size raised: {type(e).__name__}: {e}"
            continue

        if got_val_results == ref_val_results:
            val_ok += 1
        elif "_note" not in out:
            out["_note"] = f"M1 config {i} validate mismatch: got {got_val_results[:2]}, want {ref_val_results[:2]}"

        ref_filter = ref.filter_valid_block_sizes(backend, model, candidates)
        try:
            got_filter = filter_valid_block_sizes(backend, model, candidates)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"M1 config {i} filter_valid_block_sizes raised: {type(e).__name__}: {e}"
            continue

        if got_filter == ref_filter:
            filter_ok += 1
        elif "_note" not in out:
            out["_note"] = f"M1 config {i} filter mismatch: got {got_filter}, want {ref_filter}"

    if val_ok == len(ref.CONFIGS_M1):
        out["validations_matched"] = 1.0
    if filter_ok == len(ref.CONFIGS_M1):
        out["filters_matched"] = 1.0

    return out
