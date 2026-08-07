import numpy as np


def generate_reference_config():
    return {"seed": 42, "steps": 5, "batch_size": 16}


def run_oracle(config):
    from bakeoff.engine import BakeoffEngine
    engine = BakeoffEngine(config)
    return engine.run_benchmark(runs=3)
