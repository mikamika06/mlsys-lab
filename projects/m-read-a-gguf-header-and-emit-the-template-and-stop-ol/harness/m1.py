import ref


def check(workdir):
    from ggufparse.header import extract_metadata
    out = {"header_matched": 0.0}
    try:
        res = extract_metadata(ref.SAMPLE_BYTES)
        if isinstance(res, dict) and "template" in res and "stop" in res:
            out["header_matched"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
