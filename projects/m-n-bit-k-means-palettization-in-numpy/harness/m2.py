import ref


def check(workdir):
    from palettize.bytes import exact_tensor_bytes

    out = {"bytes_match": 0.0}
    ref_b = ref.exact_tensor_bytes(1024, 4, vector_length=2, centroid_dtype_bytes=4)
    try:
        got_b = exact_tensor_bytes(1024, 4, vector_length=2, centroid_dtype_bytes=4)
        if got_b == ref_b:
            out["bytes_match"] = 1.0
        else:
            out["_note"] = f"expected {ref_b} bytes, got {got_b}"
    except Exception as e:
        out["_note"] = str(e)
    return out
