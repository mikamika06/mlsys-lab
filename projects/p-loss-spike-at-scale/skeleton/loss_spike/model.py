class LossModel:
    def __init__(self, config):
        raise NotImplementedError

    def step(self, batch):
        raise NotImplementedError
