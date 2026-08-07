import ref

def check(workdir):
    from loss_spike import LossModel
    m = {"replayed_ok": 0.0}
    model = LossModel({"world_size": 64})
    logs = ref.generate_replay_logs()
    losses = [model.step(batch) for batch in logs]
    if len(losses) == 10:
        m["replayed_ok"] = 1.0
    return m
