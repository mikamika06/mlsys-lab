from .calculator import MemoryCalculator

class PeakPredictor:
    def __init__(self, calculator_cls=None):
        self.calculator_cls = calculator_cls or MemoryCalculator

    def predict_peak_bytes(self, config: dict) -> float:
        calc = self.calculator_cls(config)
        return calc.compute_total_peak_memory()

    def evaluate_error(self, config: dict, measured_bytes: float) -> float:
        pred = self.predict_peak_bytes(config)
        if measured_bytes == 0:
            return 0.0
        return abs(pred - measured_bytes) / measured_bytes
