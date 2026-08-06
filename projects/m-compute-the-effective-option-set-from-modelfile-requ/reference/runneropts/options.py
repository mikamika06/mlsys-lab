def compute_options(modelfile, request_opts, env):
    defaults = {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048, "num_gpu": 1, "seed": 42, "repeat_penalty": 1.1}
    env_map = {"LLAMA_TEMP": "temperature", "LLAMA_CTX": "num_ctx", "LLAMA_GPU": "num_gpu"}
    res = dict(defaults)
    for k, v in modelfile.items():
        res[k] = v
    for env_k, opt_k in env_map.items():
        if env_k in env:
            res[opt_k] = type(res[opt_k])(env[env_k])
    for k, v in request_opts.items():
        res[k] = v
    return res
