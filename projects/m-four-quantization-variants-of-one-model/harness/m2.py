import ref


def check(workdir):
    from edgequant.int8io import export_int8_io
    spec = ref.get_model_spec()
    res = export_int8_io(spec)
    out = {"int8_io_matched": 0.0, "size_ratio_valid": 0.0}
    if not isinstance(res, dict):
        out["_note"] = "export_int8_io did not return a dict"
        return out
    if ref.verify_int8_io(res):
        out["int8_io_matched"] = 1.0
    fp32_size = sum(w.nbytes for w in spec.values())
    int8_size = res.get("size", fp32_size)
    ratio = fp32_size / max(1, int8_size)
    if ratio >= 3.0:
        out["size_ratio_valid"] = 1.0
    else:
        out["_note"] = f"size ratio {ratio} below expected threshold 3.0"
    return out
