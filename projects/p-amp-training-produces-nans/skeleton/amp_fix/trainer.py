class SensitiveModelTrainer:
    def __init__(self, model):
        raise NotImplementedError

    def train_steps(self, data_stream, num_steps):
        raise NotImplementedError
