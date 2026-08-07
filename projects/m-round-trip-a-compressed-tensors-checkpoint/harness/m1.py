import os
import ref


def check(workdir):
    from compress.checkpoint import load_checkpoint, save_checkpoint

    out = {"parsed_correctly": 0.0}
    path = os.path.join(workdir, "test_ckpt.pkl")
    try:
        save_checkpoint(ref.SYNTHETIC_CHECKPOINT, path)
        loaded = load_checkpoint(path)
        if isinstance(loaded, dict) and "weight.quantized" in loaded:
            out["parsed_correctly"] = 1.0
    except Exception as e:
        out["_note"] = f"Failed to load checkpoint: {e}"
    finally:
        if os.path.exists(path):
            os.remove(path)
    return out
