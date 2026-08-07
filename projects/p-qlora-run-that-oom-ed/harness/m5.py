import ref

def check(workdir):
    m = {"run_completed": 0.0, "effective_batch_preserved": 0.0}
    try:
        from qlora_fix.optimizer import run_training_step
        res = run_training_step(None, 4, 4)
        if res.get("completed") is True and res.get("effective_batch") == 16:
            m["run_completed"] = 1.0
            m["effective_batch_preserved"] = 1.0
    except Exception:
        pass
    return m
