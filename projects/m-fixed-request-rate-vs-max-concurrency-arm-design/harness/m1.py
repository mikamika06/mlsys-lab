import ref

def check(workdir):
    from bench.parser import parse_vllm_json
    out = {"configs_matched": 0.0}
    ok = 0
    for raw in ref.SAMPLE_JSONS:
        want = ref.parse_vllm_json(raw)
        try:
            got = parse_vllm_json(raw)
            if got == want:
                ok += 1
        except Exception:
            pass
    out["configs_matched"] = float(ok)
    return out
