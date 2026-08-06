import torch


class CapturedStep:
    def __init__(self, model, optimizer, loss_fn):
        raise NotImplementedError

    def capture(self, sample_inputs, sample_targets):
        raise NotImplementedError

    def replay(self, inputs, targets):
        raise NotImplementedError
