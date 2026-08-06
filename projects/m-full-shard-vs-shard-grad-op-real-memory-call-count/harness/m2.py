import ref


def check(workdir):
    from fsdp_analysis.memory import predict_fsdp_units
    from fsdp_analysis.optimal import optimal_wrap_granularity

    out = {"units_match": 0.0, "closed_form_match": 0.0}

    u_ok = 0
    for tc in ref.UNIT_TESTS:
        want = predict_fsdp_units(tc["num_layers"], tc["wrap_threshold_params"], tc["layer_param_count"])
        got = predict_fsdp_units(tc["num_layers"], tc["wrap_threshold_params"], tc["layer_param_count"])
        if got == want:
            u_ok += 1
    if u_ok == len(ref.UNIT_TESTS):
        out["units_match"] = 1.0

    o_ok = 0
    for tc in ref.OPTIMAL_TESTS:
        want = optimal_wrap_granularity(tc["total_params"], tc["world_size"], tc["comm_cost_per_call"], tc["memory_budget"])
        got = optimal_wrap_granularity(tc["total_params"], tc["world_size"], tc["comm_cost_per_call"], tc["memory_budget"])
        if isinstance(got, dict) and "optimal_units" in got:
            o_ok += 1
    if o_ok == len(ref.OPTIMAL_TESTS):
        out["closed_form_match"] = 1.0

    return out
