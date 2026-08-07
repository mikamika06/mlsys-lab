def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from planner.calculator import MemoryPlanner

    m = {"zero1_states": 0.0, "zero2_grads": 0.0, "zero3_weights": 0.0, "offload_states": 0.0}

    cfg0 = {"num_params": 1000, "world_size": 4, "zero_stage": 0, "cpu_offload": False}
    p0 = MemoryPlanner(cfg0)

    cfg1 = dict(cfg0, zero_stage=1)
    p1 = MemoryPlanner(cfg1)
    if p1.opt_states_memory() * 4 == p0.opt_states_memory() and p1.grads_memory() == p0.grads_memory():
        m["zero1_states"] = 1.0

    cfg2 = dict(cfg0, zero_stage=2)
    p2 = MemoryPlanner(cfg2)
    if p2.grads_memory() * 4 == p0.grads_memory() and p2.weights_memory() == p0.weights_memory():
        m["zero2_grads"] = 1.0

    cfg3 = dict(cfg0, zero_stage=3)
    p3 = MemoryPlanner(cfg3)
    if p3.weights_memory() * 4 == p0.weights_memory():
        m["zero3_weights"] = 1.0

    cfgo = dict(cfg0, cpu_offload=True)
    po = MemoryPlanner(cfgo)
    if po.opt_states_memory() == 0:
        m["offload_states"] = 1.0

    return m
