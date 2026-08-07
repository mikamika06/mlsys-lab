import math
from runner.engine import EngineConfig

class QueueModel:
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()

    def predict_p95_latency(self, num_users: int, prompt_len: int, output_len: int) -> float:
        max_slots = max(1, int(self.config.gpu_memory_mb // self.config.bytes_per_slot_mb))

        def service_time(concurrent_count: int) -> float:
            c = min(concurrent_count, self.config.max_batch_size)
            p_time = prompt_len * self.config.prefill_ms_per_tok
            d_time = output_len * (self.config.decode_base_ms + c * self.config.decode_per_slot_ms)
            return p_time + d_time

        if num_users <= max_slots:
            return service_time(num_users)

        full_batches = (num_users - 1) // max_slots
        rem = num_users - full_batches * max_slots

        queue_wait = full_batches * service_time(max_slots)
        last_batch_service = service_time(rem)

        return queue_wait + last_batch_service
