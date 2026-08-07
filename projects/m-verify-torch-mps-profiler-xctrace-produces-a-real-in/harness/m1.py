import ref


def check(workdir):
    from msprofiler.trace import parse_signposts

    out = {"signposts_matched": 0.0}
    try:
        parsed = parse_signposts(ref.SAMPLE_XML)
        if not isinstance(parsed, list):
            out["_note"] = "parse_signposts did not return a list"
            return out

        expected = parse_signposts.__globals__["__file__"] if hasattr(parse_signposts, "__globals__") else ""
        if len(parsed) == 3 and parsed[0]["name"] == "DispatchKernel":
            out["signposts_matched"] = 3.0
        else:
            out["_note"] = f"Parsed incorrect signposts structure: {parsed}"
    except Exception as e:
        out["_note"] = f"Error during signpost parsing: {type(e).__name__}: {str(e)[:100]}"
    return out
