def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    try:
        from planner.calculator import MemoryPlanner
    except ImportError:
        return {"formulas_weights": 0.0, "formulas_grads": 0.0, "formulas_states": 0.0, "formulas_activations": 0.0}

    m = {"formulas_weights": 0.0, "formulas_grads": 0.0, "formulas_states": 0.0, "formulas_activations": 0.0}
    cfg = {"num_params": 1000, "bytes_per_param": 2, "zero_stage": 0, "world_size": 1, "cpu_offload": False, "hidden_size": 10, "num_layers": 5, "seq_len": 20, "micro_batch_size": 2, "activation_checkpointing": False}
    p = MemoryPlanner(cfg)

    if p.weights_memory() == ref.oracle_weights(cfg):
        m["formulas_weights"] = 1.0
    if p.grads_memory() == ref.oracle_grads(cfg):
        m["formulas_grads"] = 1.0
    if p.opt_states_memory() == ref.oracle_opt_states(cfg):
        m["formulas_states"] = 1.0
    if p.activations_memory() == ref.oracle_activations(cfg):
        m["formulas_activations"] = 1.0
    return m
