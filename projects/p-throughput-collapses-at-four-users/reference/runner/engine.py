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
        self.config = config or EngineConfig()

    def run_trace(self, requests: list[Request]) -> list[RequestMetrics]:
        max_slots = max(1, int(self.config.gpu_memory_mb // self.config.bytes_per_slot_mb))
        sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
        metrics = []

        pending = list(sorted_reqs)
        active = []
        current_time = 0.0

        while pending or active:
            if not active and pending and pending[0].arrival_time > current_time:
                current_time = pending[0].arrival_time

            while pending and pending[0].arrival_time <= current_time and len(active) < max_slots:
                req = pending.pop(0)
                start_t = current_time
                prefill_dur = req.prompt_len * self.config.prefill_ms_per_tok
                prefill_end_t = start_t + prefill_dur
                active.append({
                    "req": req,
                    "start_time": start_t,
                    "prefill_end_time": prefill_end_t,
                    "tokens_left": req.output_len,
                    "phase": "prefill" if prefill_dur > 0 else "decode"
                })

            if not active:
                continue

            all_prefilling = all(item["phase"] == "prefill" for item in active)
            if all_prefilling:
                next_event_time = min(item["prefill_end_time"] for item in active)
                current_time = next_event_time
                for item in active:
                    if item["prefill_end_time"] <= current_time:
                        item["phase"] = "decode"
            else:
                num_decoding = sum(1 for item in active if item["phase"] == "decode")
                step_dur = self.config.decode_base_ms + num_decoding * self.config.decode_per_slot_ms
                current_time += step_dur

                finished_indices = []
                for idx, item in enumerate(active):
                    if item["phase"] == "prefill":
                        if item["prefill_end_time"] <= current_time:
                            item["phase"] = "decode"
                    else:
                        item["tokens_left"] -= 1
                        if item["tokens_left"] <= 0:
                            finished_indices.append(idx)

                for idx in sorted(finished_indices, reverse=True):
                    item = active.pop(idx)
                    req = item["req"]
                    finish_t = current_time
                    start_t = item["start_time"]
                    p_end_t = item["prefill_end_time"]
                    q_time = start_t - req.arrival_time
                    p_time = p_end_t - start_t
                    d_time = finish_t - p_end_t
                    tot_time = finish_t - req.arrival_time
                    tps = (req.output_len / (tot_time / 1000.0)) if tot_time > 0 else 0.0
                    metrics.append(RequestMetrics(
                        req_id=req.req_id,
                        arrival_time=req.arrival_time,
                        start_time=start_t,
                        prefill_end_time=p_end_t,
                        finish_time=finish_t,
                        queue_time_ms=q_time,
                        prefill_time_ms=p_time,
                        decode_time_ms=d_time,
                        total_time_ms=tot_time,
                        tokens_generated=req.output_len,
                        tok_per_sec=tps
                    ))

        return metrics
