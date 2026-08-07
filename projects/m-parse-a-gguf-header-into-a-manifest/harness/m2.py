import ref


def check(workdir):
    from gguf_parser.overhead import compute_container_overhead

    out = {
        "overhead_matched": 0.0,
        "waste_matched": 0.0,
    }

    overhead_ok = True
    waste_ok = True

    for i, fix in enumerate(ref.GENERATED_FIXTURES):
        bin_data = fix["binary"]
        try:
            got = compute_container_overhead(bin_data)
            expected = ref.compute_container_overhead(bin_data)

            for key in [
                "header_size",
                "data_offset",
                "header_padding",
                "total_overhead",
                "raw_tensor_bytes",
                "total_file_bytes",
            ]:
                if got.get(key) != expected.get(key):
                    overhead_ok = False
                    if "_note" not in out:
                        out["_note"] = (
                            f"Fixture {i} key {key} mismatch: got {got.get(key)}, expected {expected.get(key)}"
                        )

            if got.get("alignment_waste") != expected.get("alignment_waste"):
                waste_ok = False
                if "_note" not in out:
                    out["_note"] = (
                        f"Fixture {i} alignment_waste mismatch: got {got.get('alignment_waste')}, expected {expected.get('alignment_waste')}"
                    )
        except Exception as e:
            overhead_ok = False
            waste_ok = False
            if "_note" not in out:
                out["_note"] = f"Fixture {i} raised {type(e).__name__}: {str(e)}"

    out["overhead_matched"] = 1.0 if overhead_ok else 0.0
    out["waste_matched"] = 1.0 if waste_ok else 0.0
    return out
