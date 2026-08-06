import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from safetensors_interop.compare import (
        parse_gguf_bytes,
        verify_f16_bit_identity,
    )

    out = {"gguf_parsed": 0.0, "bit_identical_matches": 0.0}

    gguf_bytes = ref.make_gguf(ref.TENSORS_1)
    st_bytes = ref.make_safetensors(ref.TENSORS_1)

    try:
        gguf_tensors = parse_gguf_bytes(gguf_bytes)
    except Exception as e:
        out["_note"] = f"parse_gguf_bytes raised {type(e).__name__}: {e}"
        return out

    gguf_ok = True
    for name, arr in ref.TENSORS_1.items():
        if name not in gguf_tensors:
            gguf_ok = False
            out["_note"] = f"Missing tensor {name} in GGUF parsed output"
            break
        parsed_arr = gguf_tensors[name]["array"]
        if not np.array_equal(parsed_arr, arr):
            gguf_ok = False
            out["_note"] = f"GGUF array value mismatch for {name}"
            break

    if gguf_ok:
        out["gguf_parsed"] = 1.0

    try:
        res = verify_f16_bit_identity(st_bytes, gguf_bytes)
    except Exception as e:
        out["_note"] = f"verify_f16_bit_identity raised {type(e).__name__}: {e}"
        return out

    if (
        isinstance(res, dict)
        and res.get("bit_identical") is True
        and len(res.get("matched_tensors", [])) == len(ref.TENSORS_1)
    ):
        out["bit_identical_matches"] = 1.0
    else:
        out["_note"] = (
            f"verify_f16_bit_identity returned unexpected result: {res}"
        )

    return out
