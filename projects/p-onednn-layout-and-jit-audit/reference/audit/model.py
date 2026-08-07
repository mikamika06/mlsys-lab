class ModelSequence:
    def __init__(self, layers):
        self.layers = list(layers)

    def reorder_layers(self, new_order):
        self.layers = [self.layers[i] for i in new_order]

    def profile(self):
        return {"time_ms": 10.0 if any("reorder" in str(l)) else 3.0}
