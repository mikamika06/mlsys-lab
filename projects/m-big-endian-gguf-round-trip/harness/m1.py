import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref

    out = {"metadata_exact": 0.0, "byte_exact_fraction": 0.0}

    try:
        from gguf_be.writer import write_gguf_be
        from gguf_be.reader import read_gguf_be
    except Exception as e:
        out["_note"] = f"Failed to import gguf_be: {type(e).__name__}: {e}"
        return out

    total_cases = len(ref.TEST_CASES_M1)
    matched_cases = 0
    byte_exact_sum = 0.0

    for cfg in ref.TEST_CASES_M1:
        want_buf = ref.ref_write_gguf_be(cfg, [], cfg.get("general.alignment", 32))
        try:
            got_buf = write_gguf_be(cfg, [], cfg.get("general.alignment", 32))
        except Exception as e:
            out["_note"] = f"write_gguf_be failed: {e}"
            return out

        byte_exact_sum += ref.compute_byte_exact_fraction(got_buf, want_buf)

        try:
            got_meta, _, _ = read_gguf_be(got_buf)
        except Exception as e:
            out["_note"] = f"read_gguf_be failed: {e}"
            return out

        if all(
            k in got_meta and ref.compare_vals(got_meta[k], v) for k, v in cfg.items()
        ):
            matched_cases += 1

    out["metadata_exact"] = float(matched_cases) / float(total_cases)
    out["byte_exact_fraction"] = byte_exact_sum / float(total_cases)
    return out
