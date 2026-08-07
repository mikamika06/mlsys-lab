class AutoRoundModifier:
    def __init__(self, steps=100, lr=0.1, bits=4):
        raise NotImplementedError

    def fit_transform(self, layers, calibration_data):
        raise NotImplementedError
