import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"eager_matched": 0.0, "sdpa_matched": 0.0}

    try:
        from memacc.accounting import layer_eager_memory, layer_sdpa_memory
    except Exception as e:
        out["_note"] = f"Failed to import memacc.accounting: {e}"
        return out

    eager_ok = True
    sdpa_ok = True

    for cfg in ref.TEST_LAYERS:
        want_eager = ref.layer_eager_memory(cfg)
        try:
            got_eager = layer_eager_memory(cfg)
            if got_eager != want_eager:
                eager_ok = False
                out["_note"] = f"Eager mismatch for layer {cfg}: got {got_eager}, expected {want_eager}"
                break
        except Exception as e:
            eager_ok = False
            out["_note"] = f"Error calling layer_eager_memory: {e}"
            break

        want_sdpa = ref.layer_sdpa_memory(cfg)
        try:
            got_sdpa = layer_sdpa_memory(cfg)
            if got_sdpa != want_sdpa:
                sdpa_ok = False
                out["_note"] = f"SDPA mismatch for layer {cfg}: got {got_sdpa}, expected {want_sdpa}"
                break
        except Exception as e:
            sdpa_ok = False
            out["_note"] = f"Error calling layer_sdpa_memory: {e}"
            break

    if eager_ok:
        out["eager_matched"] = 1.0
    if sdpa_ok:
        out["sdpa_matched"] = 1.0

    return out
