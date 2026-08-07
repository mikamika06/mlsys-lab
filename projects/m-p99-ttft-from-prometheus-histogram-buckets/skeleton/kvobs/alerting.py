class HysteresisAlert:
    """Stateful alert evaluator with hysteresis and hold counts."""

    def __init__(self, high_threshold: float, low_threshold: float, hold_periods: int = 3):
        raise NotImplementedError

    def process(self, value: float) -> bool:
        raise NotImplementedError
