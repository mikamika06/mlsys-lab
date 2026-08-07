import hashlib
from runner.config import merge_options

class ChatClient:
    def __init__(self, default_options=None):
        self.default_options = default_options or {"seed": 42, "temperature": 0.0, "num_predict": 128}

    def prepare_payload(self, prompt, req_options=None):
        opts = merge_options(self.default_options, {}, req_options or {})
        return {
            "prompt": prompt,
            "options": opts
        }

    def generate(self, prompt, seed=42, temperature=0.0, num_predict=128):
        req_opts = {"seed": seed, "temperature": temperature, "num_predict": num_predict}
        payload = self.prepare_payload(prompt, req_opts)
        s = payload["options"]["seed"]
        t = payload["options"]["temperature"]
        np = payload["options"]["num_predict"]
        h = hashlib.sha256(f"{prompt}_{s}_{t}_{np}".encode()).hexdigest()
        return f"output_{h[:16]}"
