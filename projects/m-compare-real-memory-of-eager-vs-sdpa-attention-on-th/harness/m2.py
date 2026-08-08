import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"model_bytes_matched": 0.0, "size_ratio": 0.0}

    try:
        from memacc.compare import compute_activation_size_ratio, model_total_retained_bytes
    except Exception as e:
        out["_note"] = f"Failed to import memacc.compare: {e}"
        return out

    model_cfg = ref.SAMPLE_MODEL

    try:
        got_eager = model_total_retained_bytes(model_cfg, "eager")
        want_eager = ref.model_total_retained_bytes(model_cfg, "eager")
        got_sdpa = model_total_retained_bytes(model_cfg, "sdpa")
        want_sdpa = ref.model_total_retained_bytes(model_cfg, "sdpa")

        if got_eager == want_eager and got_sdpa == want_sdpa:
            out["model_bytes_matched"] = 1.0
        else:
            out["_note"] = (
                f"Model bytes mismatch. Eager: got {got_eager}, want {want_eager}. "
                f"SDPA: got {got_sdpa}, want {want_sdpa}."
            )
            return out
    except Exception as e:
        out["_note"] = f"Error executing model_total_retained_bytes: {e}"
        return out

    try:
        got_ratio = compute_activation_size_ratio(model_cfg)
        out["size_ratio"] = float(got_ratio)
    except Exception as e:
        out["_note"] = f"Error executing compute_activation_size_ratio: {e}"

    return out
