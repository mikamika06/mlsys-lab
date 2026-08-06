import ref

def check(workdir):
    from vllm_engine.parser import parse_argv
    data = ref.generate_test_cases()
    out = {"parser_matched": 0.0}
    try:
        got = parse_argv(data["argv"])
        if got == data["parsed_argv"]:
            out["parser_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {data['parsed_argv']}"
    except Exception as e:
        out["_note"] = f"exception: {type(e).__name__}: {e}"
    return out
