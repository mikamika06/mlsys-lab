class ModelSequence:
    def __init__(self, layers):
        raise NotImplementedError

    def reorder_layers(self, new_order):
        raise NotImplementedError

    def profile(self):
        raise NotImplementedError
