import ref

def check(workdir):
    from fsdpmeasure.residency import get_parameter_residency
    model_size = 1024 * 1024 * 100
    want_full = ref.get_parameter_residency(model_size, "FULL_SHARD", "between_forward")
    want_grad = ref.get_parameter_residency(model_size, "SHARD_GRAD_OP", "between_forward")

    got_full = get_parameter_residency(model_size, "FULL_SHARD", "between_forward")
    got_grad = get_parameter_residency(model_size, "SHARD_GRAD_OP", "between_forward")

    match = 1.0 if got_full == want_full and got_grad == want_grad else 0.0
    return {"residency_match": match}
