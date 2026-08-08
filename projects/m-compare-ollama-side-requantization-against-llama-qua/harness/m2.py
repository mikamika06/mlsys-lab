import ref


def check(workdir):
    from ollamabridge.import_model import verify_safetensors_dir
    out = {"arch_matched": 0.0}
    try:
        ok1 = verify_safetensors_dir({"architecture": "llama"})
        ok2 = verify_safetensors_dir({"architecture": "unsupported"})
        if ok1 is True and ok2 is False:
            out["arch_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"m2 failed: {type(e).__name__}: {str(e)[:100]}"
    return out
