import numpy as np
from quantizer.params import calc_affine_params, quantize


class CalibrationDataReader:
    def __init__(self, data_batches: list[dict[str, np.ndarray]]):
        self._data = data_batches
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._data):
            return None
        batch = self._data[self._index]
        self._index += 1
        return batch

    def reset(self) -> None:
        self._index = 0


def calibrate_static_params(reader: CalibrationDataReader, qmin: int = 0, qmax: int = 255) -> dict[str, tuple[float, int]]:
    reader.reset()
    stats = {}
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        for name, arr in batch.items():
            b_min = float(np.min(arr))
            b_max = float(np.max(arr))
            if name not in stats:
                stats[name] = [b_min, b_max]
            else:
                stats[name][0] = min(stats[name][0], b_min)
                stats[name][1] = max(stats[name][1], b_max)

    params = {}
    for name, (g_min, g_max) in stats.items():
        params[name] = calc_affine_params(g_min, g_max, qmin, qmax)
    return params


def quantize_dataset_static(reader: CalibrationDataReader, params: dict[str, tuple[float, int]], qmin: int = 0, qmax: int = 255) -> dict[str, list[np.ndarray]]:
    reader.reset()
    res = {}
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        for name, arr in batch.items():
            scale, zp = params[name]
            q_arr = quantize(arr, scale, zp, qmin, qmax)
            if name not in res:
                res[name] = []
            res[name].append(q_arr)
    return res
