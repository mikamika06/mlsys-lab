class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 65536.0,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 65536.0 * 65536.0,
    ):
        self.scale = float(init_scale)
        self.scale_factor = float(scale_factor)
        self.scale_window = int(scale_window)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.consecutive_good_steps = 0

    def update(self, has_overflow: bool) -> float:
        if has_overflow:
            self.consecutive_good_steps = 0
            self.scale = max(self.min_scale, self.scale / self.scale_factor)
        else:
            self.consecutive_good_steps += 1
            if self.consecutive_good_steps == self.scale_window:
                self.scale = min(self.max_scale, self.scale * self.scale_factor)
                self.consecutive_good_steps = 0
        return float(self.scale)


def simulate_trajectory(
    init_scale: float,
    scale_factor: float,
    scale_window: int,
    min_scale: float,
    max_scale: float,
    overflow_sequence: list[bool],
) -> list[float]:
    scaler = DynamicLossScaler(
        init_scale=init_scale,
        scale_factor=scale_factor,
        scale_window=scale_window,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    trajectory = []
    for overflow in overflow_sequence:
        s = scaler.update(overflow)
        trajectory.append(s)
    return trajectory
