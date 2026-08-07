import numpy as np


class PrefillWorker:
    def __init__(self, gpu_id, num_layers, head_dim, num_heads):
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.flushed_kv = {}
        self.stats = {"prefill_flops": 0, "prefill_steps": 0}

    def process_prefill(self, request_id, prompt_tokens):
        seq_len = len(prompt_tokens)
        flops = 2 * self.num_layers * seq_len * seq_len * self.num_heads * self.head_dim
        self.stats["prefill_flops"] += flops
        self.stats["prefill_steps"] += 1

        kv_payload = {
            "request_id": request_id,
            "seq_len": seq_len,
            "blocks": np.ones((self.num_layers, 2, seq_len, self.num_heads, self.head_dim), dtype=np.float32)
        }
        self.flushed_kv[request_id] = kv_payload
        return kv_payload


class DecodeWorker:
    def __init__(self, gpu_id, num_layers, head_dim, num_heads):
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.kv_store = {}
        self.stats = {"prefill_flops": 0, "prefill_steps": 0, "decode_flops": 0, "decode_steps": 0}

    def receive_kv_cache(self, request_id, kv_payload):
        self.kv_store[request_id] = kv_payload

    def step_decode(self, request_id, next_token):
        if request_id not in self.kv_store:
            fake_prompt_len = 128
            self.stats["prefill_flops"] += 2 * self.num_layers * fake_prompt_len * fake_prompt_len * self.num_heads * self.head_dim
            self.stats["prefill_steps"] += 1
            curr_len = fake_prompt_len
        else:
            curr_len = self.kv_store[request_id]["seq_len"]

        flops = 2 * self.num_layers * 1 * curr_len * self.num_heads * self.head_dim
        self.stats["decode_flops"] += flops
        self.stats["decode_steps"] += 1
        self.kv_store[request_id]["seq_len"] = curr_len + 1
        return next_token + 1


class Pipeline1P1D:
    def __init__(self, prefill_worker, decode_worker):
        self.prefill_worker = prefill_worker
        self.decode_worker = decode_worker

    def process_request(self, request_id, prompt_tokens, decode_steps):
        kv_payload = self.prefill_worker.process_prefill(request_id, prompt_tokens)
        self.decode_worker.receive_kv_cache(request_id, kv_payload)

        curr_token = prompt_tokens[-1]
        tokens_out = []
        for _ in range(decode_steps):
            curr_token = self.decode_worker.step_decode(request_id, curr_token)
            tokens_out.append(curr_token)

        return {
            "request_id": request_id,
            "tokens": tokens_out,
            "prefill_stats": dict(self.prefill_worker.stats),
            "decode_stats": dict(self.decode_worker.stats)
        }
