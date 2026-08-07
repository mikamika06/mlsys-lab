def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from planner.calculator import MemoryPlanner

    m = {"checkpointing_reduces": 0.0, "micro_batch_scales": 0.0}
    cfg = {"hidden_size": 10, "num_layers": 10, "seq_len": 20, "micro_batch_size": 2, "activation_checkpointing": False}
    p1 = MemoryPlanner(cfg)
    cfg["activation_checkpointing"] = True
    p2 = MemoryPlanner(cfg)

    if p2.activations_memory() * 10 == p1.activations_memory():
        m["checkpointing_reduces"] = 1.0

    cfg["micro_batch_size"] = 1
    p3 = MemoryPlanner(cfg)
    if p3.activations_memory() * 2 == p2.activations_memory():
        m["micro_batch_scales"] = 1.0

    return m
