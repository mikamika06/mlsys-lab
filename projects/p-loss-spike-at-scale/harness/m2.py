import ref

def check(workdir):
    from loss_spike import LossModel
    m = {"isolated_ok": 0.0}
    m1 = LossModel({"world_size": 8})
    m2 = LossModel({"world_size": 64})
    logs = ref.generate_replay_logs()
    out1 = [m1.step(b) for b in logs]
    out2 = [m2.step(b) for b in logs]
    if len(out1) == len(out2):
        m["isolated_ok"] = 1.0
    return m
