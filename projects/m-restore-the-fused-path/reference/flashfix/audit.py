def triage_warnings(logs):
    out = []
    for log in logs:
        if "non-contiguous" in log:
            out.append("layout")
        elif "stride mismatch" in log:
            out.append("stride")
        elif "head dimension" in log:
            out.append("alignment")
        else:
            out.append("unknown")
    return out
