class HysteresisAlert:
    """Stateful alert evaluator with hysteresis and hold counts."""

    def __init__(self, high_threshold: float, low_threshold: float, hold_periods: int = 3):
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.hold_periods = int(hold_periods)
        self.firing = False
        self.high_count = 0
        self.low_count = 0

    def process(self, value: float) -> bool:
        val = float(value)
        if val >= self.high_threshold:
            self.high_count += 1
            self.low_count = 0
            if self.high_count >= self.hold_periods:
                self.firing = True
        elif val < self.low_threshold:
            self.low_count += 1
            self.high_count = 0
            if self.low_count >= self.hold_periods:
                self.firing = False
        else:
            self.high_count = 0
            self.low_count = 0

        return self.firing
