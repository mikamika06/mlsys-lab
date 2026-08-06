import ref

def check(workdir):
    from vllm_engine.resolver import resolve_args
    data = ref.generate_test_cases()
    out = {"precedence_matched": 0.0}
    try:
        got = resolve_args(data["defaults"], data["yaml_cfg"], data["env_cfg"], data["cli_cfg"])
        if got == data["resolved"]:
            out["precedence_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {data['resolved']}"
    except Exception as e:
        out["_note"] = f"exception: {type(e).__name__}: {e}"
    return out
