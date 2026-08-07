import numpy as np


class BakeoffEngine:
    def __init__(self, config):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def step(self, backend_id):
        raise NotImplementedError

    def get_weights(self, backend_id):
        raise NotImplementedError

    def evaluate(self, backend_id):
        raise NotImplementedError

    def run_benchmark(self, runs=3):
        raise NotImplementedError

    def recommend(self):
        raise NotImplementedError
