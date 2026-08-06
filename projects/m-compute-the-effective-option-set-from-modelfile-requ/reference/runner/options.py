"""Option precedence and classification logic."""


def compute_effective_options(modelfile_params, request_options, env_vars):
    """Compute effective options enforcing request > Modelfile > env precedence."""
    effective = {}
    if env_vars:
        for k, v in env_vars.items():
            if k.startswith("RUNNER_OPTION_"):
                opt_key = k[len("RUNNER_OPTION_") :].lower()
                effective[opt_key] = v
            elif k in ("NUM_GPU", "NUM_THREAD", "CONTEXT_SIZE"):
                effective[k.lower()] = v

    if modelfile_params:
        for k, v in modelfile_params.items():
            effective[k.lower()] = v

    if request_options:
        for k, v in request_options.items():
            effective[k.lower()] = v

    return effective


def classify_options(options, executor_fn):
    """Partition options into load-time and sample-time by checking load_duration."""
    base_res = executor_fn({})
    base_duration = base_res.get("load_duration", 0)

    load_time = []
    sample_time = []

    for opt in options:
        res = executor_fn({opt: 999})
        duration = res.get("load_duration", 0)
        if duration > base_duration:
            load_time.append(opt)
        else:
            sample_time.append(opt)

    return {"load_time": sorted(load_time), "sample_time": sorted(sample_time)}
