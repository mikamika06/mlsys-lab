import ref


def check(workdir):
    from triton_bench.analysis import parse_benchmark

    out = {"parsed_correctly": 0.0}
    try:
        res = parse_benchmark(ref.RAW_DATA)
        if isinstance(res, list) and len(res) == len(ref.RAW_DATA):
            if res[0]["size"] == ref.RAW_DATA[0]["size"]:
                out["parsed_correctly"] = 1.0
    except Exception as e:
        out["_note"] = f"error: {str(e)[:100]}"
    return out
