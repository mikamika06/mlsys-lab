import ref

def check(workdir):
    from runneropts.options import compute_options
    cases = ref.get_test_cases()
    matched = 0
    for i, (mf, req, env) in enumerate(cases):
        defaults = {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048, "num_gpu": 1, "seed": 42, "repeat_penalty": 1.1}
        env_map = {"LLAMA_TEMP": "temperature", "LLAMA_CTX": "num_ctx", "LLAMA_GPU": "num_gpu"}
        expected = dict(defaults)
        for k, v in mf.items():
            expected[k] = v
        for env_k, opt_k in env_map.items():
            if env_k in env:
                expected[opt_k] = type(expected[opt_k])(env[env_k])
        for k, v in req.items():
            expected[k] = v
        try:
            got = compute_options(mf, req, env)
            if got == expected:
                matched += 1
        except Exception:
            pass
    return {"options_matched": float(matched)}
