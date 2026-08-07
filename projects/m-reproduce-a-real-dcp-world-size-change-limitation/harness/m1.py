import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from dcpfix.parser import parse_dcp_metadata

    metadata, _, _ = ref.generate_test_case(seed=100)
    out = {"metadata_matched": 0.0}
    try:
        parsed = parse_dcp_metadata(metadata)
        if isinstance(parsed, dict) and "model.weight" in parsed:
            if parsed["model.weight"]["shape"] == [4, 4]:
                out["metadata_matched"] = 3.0
        if out["metadata_matched"] == 0.0:
            out["_note"] = f"Parsed metadata structure incorrect: {parsed}"
    except Exception as e:
        out["_note"] = f"Parser raised exception: {type(e).__name__}: {str(e)}"
    return out
