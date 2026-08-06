import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from safetensors_interop.header import parse_safetensors_bytes

    out = {"headers_parsed": 0.0, "tensors_extracted": 0.0}

    st_bytes = ref.make_safetensors(ref.TENSORS_1)
    ref_parsed = ref.parse_safetensors_reference(st_bytes)

    try:
        got_parsed = parse_safetensors_bytes(st_bytes)
    except Exception as e:
        out["_note"] = f"parse_safetensors_bytes raised {type(e).__name__}: {e}"
        return out

    header_ok = True
    for name, ref_meta in ref_parsed.items():
        if name not in got_parsed:
            header_ok = False
            out["_note"] = f"Missing tensor {name} in parsed header"
            break
        got_meta = got_parsed[name]
        if got_meta["dtype"] != ref_meta["dtype"] or list(
            got_meta["shape"]
        ) != list(ref_meta["shape"]):
            header_ok = False
            out["_note"] = (
                f"Metadata mismatch for {name}: got {got_meta}, want {ref_meta}"
            )
            break

    if header_ok:
        out["headers_parsed"] = 1.0

    tensors_ok = True
    for name, ref_meta in ref_parsed.items():
        got_meta = got_parsed[name]
        if got_meta["data"] != ref_meta["data"]:
            tensors_ok = False
            out["_note"] = f"Raw bytes mismatch for tensor {name}"
            break
        if not np.array_equal(got_meta["array"], ref_meta["array"]):
            tensors_ok = False
            out["_note"] = f"Array values mismatch for tensor {name}"
            break

    if tensors_ok:
        out["tensors_extracted"] = 1.0

    return out
