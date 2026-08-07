import ref
from amp_fix.trainer import SensitiveModelTrainer


def check(workdir):
    m = {"sensitive_isolated": 0.0}
    class DummyModel:
        def __call__(self, x):
            return x * 2.0
    trainer = SensitiveModelTrainer(DummyModel())
    steps = trainer.train_steps([1.0, 2.0, 3.0], 3)
    if steps == 3:
        m["sensitive_isolated"] = 1.0
    return m
