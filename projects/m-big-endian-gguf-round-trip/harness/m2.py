import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref

    out = {"zero_copy_exact": 0.0, "byte_exact_fraction": 0.0}

    try:
        from gguf_be.writer import write_gguf_be
        from gguf_be.reader import read_gguf_be
        from gguf_be.zero_copy import extract_tensor_zero_copy
    except Exception as e:
        out["_note"] = f"Failed to import gguf_be: {type(e).__name__}: {e}"
        return out

    tensors = ref.generate_test_tensors()
    meta = {"model.name": "m2_model", "general.alignment": 32}

    want_buf = ref.ref_write_gguf_be(meta, tensors, 32)

    try:
        got_buf = write_gguf_be(meta, tensors, 32)
    except Exception as e:
        out["_note"] = f"write_gguf_be failed: {e}"
        return out

    out["byte_exact_fraction"] = ref.compute_byte_exact_fraction(got_buf, want_buf)

    try:
        got_meta, tensor_infos, base_offset = read_gguf_be(got_buf)
    except Exception as e:
        out["_note"] = f"read_gguf_be failed: {e}"
        return out

    if len(tensor_infos) != len(tensors):
        out["_note"] = f"Expected {len(tensors)} tensors, got {len(tensor_infos)}"
        return out

    zero_copy_ok = 0
    for t_spec, t_info in zip(tensors, tensor_infos):
        try:
            ext = extract_tensor_zero_copy(got_buf, t_info, base_offset)
        except Exception as e:
            out["_note"] = f"extract_tensor_zero_copy failed: {e}"
            return out

        expected_data = t_spec["data"]
        data_matches = ref.compare_arrays(ext, expected_data)
        is_zero_copy = (getattr(ext, "base", None) is not None) or (
            ext.flags.owndata is False
        )

        if data_matches and is_zero_copy:
            zero_copy_ok += 1

    out["zero_copy_exact"] = float(zero_copy_ok) / float(len(tensors))
    return out
