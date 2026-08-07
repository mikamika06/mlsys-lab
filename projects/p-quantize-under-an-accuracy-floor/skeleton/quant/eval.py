class Evaluator:
    """Evaluates model performance against a dataset."""

    def __init__(self, dataset):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError
