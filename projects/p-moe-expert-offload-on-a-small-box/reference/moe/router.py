class Preloader:
    def __init__(self, cache, cost_model):
        self.cache = cache
        self.cost_model = cost_model

    def predict_and_prefetch(self, routing_logits):
        top_expert = int(routing_logits.argmax())
        self.cache.access(top_expert)
        return [top_expert]
