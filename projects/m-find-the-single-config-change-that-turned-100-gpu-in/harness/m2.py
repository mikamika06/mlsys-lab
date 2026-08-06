import ref

def check(workdir):
    from offload.parser import parse_processor_column
    out = {"parsed_match": 0.0}
    s = "GPU:0\nGPU:1\nCPU\nCPU"
    got = parse_processor_column(s)
    want = ref.parse_processor_column(s)
    if got == want:
        out["parsed_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
