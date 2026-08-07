class ConditionalModel:
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def export_check(self, x):
        raise NotImplementedError
