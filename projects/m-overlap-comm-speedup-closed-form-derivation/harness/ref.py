import sys
sys.path.insert(0, ".")

OVERLAP_CASES = [
    (10.0, 10.0, 0.0, 1.0),
    (10.0, 10.0, 0.5, 1.5),
    (10.0, 10.0, 1.0, 2.0),
    (20.0, 10.0, 0.5, 1.2),
    (10.0, 30.0, 0.8, 2.5),
    (15.0, 0.0, 1.0, 1.0),
    (0.0, 20.0, 0.5, 1.5),
]

SCALER_CASES = [
    {
        "params": {
            "init_scale": 1024.0,
            "scale_factor": 2.0,
            "scale_window": 3,
            "min_scale": 1.0,
            "max_scale": 4096.0,
        },
        "seq": [False, False, False, False, True, False, False, False],
    },
    {
        "params": {
            "init_scale": 16.0,
            "scale_factor": 2.0,
            "scale_window": 2,
            "min_scale": 2.0,
            "max_scale": 32.0,
        },
        "seq": [True, True, True, True, False, False],
    },
]


def compute_step_time(t_comp: float, t_comm: float, overlap_factor: float) -> float:
    if t_comp < 0 or t_comm < 0:
        raise ValueError("Time components must be non-negative")
    overlap_factor = max(0.0, min(1.0, float(overlap_factor)))
    exposed_comm = max(0.0, t_comm - overlap_factor * t_comp)
    return float(t_comp + exposed_comm)


def compute_speedup(t_comp: float, t_comm: float, overlap_factor: float) -> float:
    unoverlapped_time = t_comp + t_comm
    if unoverlapped_time == 0:
        return 1.0
    overlapped_time = compute_step_time(t_comp, t_comm, overlap_factor)
    return float(unoverlapped_time / overlapped_time)


def min_overlap_for_speedup(t_comp: float, t_comm: float, target_speedup: float) -> float:
    if target_speedup < 1.0:
        return 0.0
    unoverlapped = t_comp + t_comm
    if unoverlapped == 0:
        return 0.0
    max_possible_speedup = unoverlapped / max(t_comp, t_comm) if max(t_comp, t_comm) > 0 else 1.0
    if target_speedup > max_possible_speedup + 1e-9:
        return -1.0
    target_step_time = unoverlapped / target_speedup
    req_exposed = target_step_time - t_comp
    if req_exposed >= t_comm:
        return 0.0
    if t_comp == 0:
        return 1.0 if req_exposed < t_comm else 0.0
    required_factor = (t_comm - req_exposed) / t_comp
    return float(max(0.0, min(1.0, required_factor)))


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


def validate_batch_config(config: dict) -> bool:
    required_keys = {
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "data_parallel_size",
    }
    if not required_keys.issubset(config.keys()):
        return False
    tbs = config["train_batch_size"]
    mbs = config["train_micro_batch_size_per_gpu"]
    gas = config["gradient_accumulation_steps"]
    dp = config["data_parallel_size"]
    if any(x <= 0 for x in (tbs, mbs, gas, dp)):
        return False
    return tbs == mbs * gas * dp


def resolve_batch_config(config: dict) -> dict:
    keys = {
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "data_parallel_size",
    }
    provided = {k: config[k] for k in keys if k in config and config[k] is not None}
    if len(provided) < 3:
        raise ValueError("At least 3 batch parameters must be provided")

    res = dict(provided)
    if len(provided) == 4:
        if not validate_batch_config(res):
            raise ValueError("Inconsistent batch configuration parameters")
        return res

    tbs = res.get("train_batch_size")
    mbs = res.get("train_micro_batch_size_per_gpu")
    gas = res.get("gradient_accumulation_steps")
    dp = res.get("data_parallel_size")

    if tbs is None:
        res["train_batch_size"] = mbs * gas * dp
    elif mbs is None:
        denom = gas * dp
        if tbs % denom != 0:
            raise ValueError("train_batch_size is not divisible by gas * dp")
        res["train_micro_batch_size_per_gpu"] = tbs // denom
    elif gas is None:
        denom = mbs * dp
        if tbs % denom != 0:
            raise ValueError("train_batch_size is not divisible by mbs * dp")
        res["gradient_accumulation_steps"] = tbs // denom
    elif dp is None:
        denom = mbs * gas
        if tbs % denom != 0:
            raise ValueError("train_batch_size is not divisible by mbs * gas")
        res["data_parallel_size"] = tbs // denom

    if not validate_batch_config(res):
        raise ValueError("Resolved configuration is invalid")

    return res
