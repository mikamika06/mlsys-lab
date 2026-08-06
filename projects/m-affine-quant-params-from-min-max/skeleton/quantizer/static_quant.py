import numpy as np


class CalibrationDataReader:
    def __init__(self, data_batches: list[dict[str, np.ndarray]]):
        raise NotImplementedError

    def get_next(self) -> dict[str, np.ndarray] | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


def calibrate_static_params(reader: CalibrationDataReader, qmin: int = 0, qmax: int = 255) -> dict[str, tuple[float, int]]:
    raise NotImplementedError


def quantize_dataset_static(reader: CalibrationDataReader, params: dict[str, tuple[float, int]], qmin: int = 0, qmax: int = 255) -> dict[str, list[np.ndarray]]:
    raise NotImplementedError
