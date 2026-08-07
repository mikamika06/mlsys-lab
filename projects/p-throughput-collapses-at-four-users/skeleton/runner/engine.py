from dataclasses import dataclass

@dataclass
class EngineConfig:
    gpu_memory_mb: int = 3072
    bytes_per_slot_mb: int = 1024
    max_batch_size: int = 4
    prefill_ms_per_tok: float = 0.5
    decode_base_ms: float = 20.0
    decode_per_slot_ms: float = 3.0

@dataclass
class Request:
    req_id: str
    arrival_time: float
    prompt_len: int
    output_len: int

@dataclass
class RequestMetrics:
    req_id: str
    arrival_time: float
    start_time: float
    prefill_end_time: float
    finish_time: float
    queue_time_ms: float
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    tokens_generated: int
    tok_per_sec: float

class Engine:
    def __init__(self, config: EngineConfig = None):
        raise NotImplementedError

    def run_trace(self, requests: list[Request]) -> list[RequestMetrics]:
        raise NotImplementedError
