class PrefillWorker:
    def __init__(self, gpu_id, num_layers, head_dim, num_heads):
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads

    def process_prefill(self, request_id, prompt_tokens):
        raise NotImplementedError

class DecodeWorker:
    def __init__(self, gpu_id, num_layers, head_dim, num_heads):
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads

    def receive_kv_cache(self, request_id, kv_payload):
        raise NotImplementedError

    def step_decode(self, request_id, next_token):
        raise NotImplementedError

class Pipeline1P1D:
    def __init__(self, prefill_worker, decode_worker):
        self.prefill_worker = prefill_worker
        self.decode_worker = decode_worker

    def process_request(self, request_id, prompt_tokens, decode_steps):
        raise NotImplementedError
