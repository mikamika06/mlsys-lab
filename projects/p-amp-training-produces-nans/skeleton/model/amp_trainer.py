def locate_first_nan(model, inputs):
    raise NotImplementedError


def check_autocast_ops(model, inputs):
    raise NotImplementedError


def simulate_grad_scaler(scaler, loss):
    raise NotImplementedError


def isolate_sensitive_layers(model):
    raise NotImplementedError


def train_stable_steps(model, dataloader, steps=1000):
    raise NotImplementedError


class NaNDetector:
    def __init__(self, model):
        raise NotImplementedError

    def register(self):
        raise NotImplementedError
