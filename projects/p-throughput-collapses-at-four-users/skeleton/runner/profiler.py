from runner.engine import EngineConfig, RequestMetrics

def build_slot_scaling_curve(engine_config: EngineConfig, slot_counts: list[int]) -> dict:
    raise NotImplementedError

def find_knee(curve_data: dict) -> int:
    raise NotImplementedError

def decompose_timing(metrics: list[RequestMetrics]) -> dict:
    raise NotImplementedError

def identify_bottleneck(engine_config: EngineConfig, active_users: int) -> str:
    raise NotImplementedError

def optimize_config(engine_config: EngineConfig) -> EngineConfig:
    raise NotImplementedError
