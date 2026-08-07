class AutoRoundModifier:
    def __init__(self, model, tokenizer=None):
        raise NotImplementedError

    def optimize(self, calibration_data):
        raise NotImplementedError
