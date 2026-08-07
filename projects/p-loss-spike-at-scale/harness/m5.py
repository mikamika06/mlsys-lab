import ref

def check(workdir):
    from loss_spike import LossModel
    m = {"converged_ok": 0.0}
    model = LossModel({"world_size": 64, "scale_factor": 0.0})
    logs = ref.generate_replay_logs()
    losses = [model.step(b) for b in logs]
    if all(l < 10.0 for l in losses):
        m["converged_ok"] = 1.0
    return m
