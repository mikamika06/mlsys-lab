from runner.engine import EngineConfig

class QueueModel:
    def __init__(self, config: EngineConfig = None):
        raise NotImplementedError

    def predict_p95_latency(self, num_users: int, prompt_len: int, output_len: int) -> float:
        raise NotImplementedError
