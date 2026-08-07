class Request:
    def __init__(self, rid: str, prompt_len: int):
        self.rid = rid
        self.prompt_len = prompt_len
        self.prefill_left = prompt_len
        self.is_decode = False

class Packer:
    def __init__(self, token_budget: int):
        self.token_budget = token_budget
        self.waiting_prefill = []
        self.active_decodes = []

    def add_request(self, req: Request):
        raise NotImplementedError

    def remove_request(self, rid: str):
        raise NotImplementedError

    def step(self) -> dict:
        """
        Returns a dictionary mapping `rid` to `tokens_processed` for this step.
        Must respect `self.token_budget`.
        Must guarantee decode progress and chunk prefill requests appropriately.
        """
        raise NotImplementedError
