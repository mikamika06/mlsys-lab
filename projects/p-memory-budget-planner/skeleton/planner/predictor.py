class PeakPredictor:
    def __init__(self, calculator_cls=None):
        raise NotImplementedError

    def predict_peak_bytes(self, config: dict) -> float:
        raise NotImplementedError

    def evaluate_error(self, config: dict, measured_bytes: float) -> float:
        raise NotImplementedError
