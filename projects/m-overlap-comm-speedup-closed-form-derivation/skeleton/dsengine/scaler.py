class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 65536.0,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 65536.0 * 65536.0,
    ):
        raise NotImplementedError

    def update(self, has_overflow: bool) -> float:
        """Update scale factor given overflow state and return new scale."""
        raise NotImplementedError


def simulate_trajectory(
    init_scale: float,
    scale_factor: float,
    scale_window: int,
    min_scale: float,
    max_scale: float,
    overflow_sequence: list[bool],
) -> list[float]:
    """Simulate loss scale values across an overflow sequence."""
    raise NotImplementedError
