import ref
from amp_fix.trainer import SensitiveModelTrainer


def check(workdir):
    m = {"stable_run": 0.0}
    class LongModel:
        def __call__(self, x):
            return x
    trainer = LongModel()
    data = [1.0] * 1000
    res_steps = SensitiveModelTrainer(trainer).train_steps(data, 1000)
    if res_steps >= 1000:
        m["stable_run"] = 1.0
    return m
