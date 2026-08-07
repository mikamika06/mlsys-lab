class Preloader:
    def __init__(self, cache, cost_model):
        raise NotImplementedError

    def predict_and_prefetch(self, routing_logits):
        raise NotImplementedError
