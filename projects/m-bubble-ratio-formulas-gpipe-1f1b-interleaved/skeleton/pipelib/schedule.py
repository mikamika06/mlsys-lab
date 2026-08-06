class ScheduleGPipe:
    """GPipe schedule simulator."""

    def __init__(self, p: int, m: int, f_cost: float = 1.0, b_cost: float = 2.0):
        raise NotImplementedError

    def run(self) -> dict:
        raise NotImplementedError


class Schedule1F1B:
    """1F1B schedule simulator."""

    def __init__(self, p: int, m: int, f_cost: float = 1.0, b_cost: float = 2.0):
        raise NotImplementedError

    def run(self) -> dict:
        raise NotImplementedError
