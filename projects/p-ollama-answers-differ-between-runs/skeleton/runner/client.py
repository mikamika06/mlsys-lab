class ChatClient:
    def __init__(self, default_options=None):
        raise NotImplementedError

    def prepare_payload(self, prompt, req_options=None):
        raise NotImplementedError

    def generate(self, prompt, seed=42, temperature=0.0, num_predict=128):
        raise NotImplementedError
