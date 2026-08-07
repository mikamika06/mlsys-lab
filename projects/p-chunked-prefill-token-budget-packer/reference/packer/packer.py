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
        self.waiting_prefill.append(req)

    def remove_request(self, rid: str):
        self.active_decodes = [r for r in self.active_decodes if r.rid != rid]

    def step(self) -> dict:
        ans = {}
        budget = self.token_budget

        new_decodes = []
        # Priority 1: decodes
        for req in self.active_decodes:
            if budget >= 1:
                ans[req.rid] = 1
                budget -= 1
                new_decodes.append(req)
            else:
                new_decodes.append(req)

        new_waiting = []
        # Priority 2: prefill chunks in FIFO
        for req in self.waiting_prefill:
            if budget > 0:
                take = min(budget, req.prefill_left)
                ans[req.rid] = take
                req.prefill_left -= take
                budget -= take

                if req.prefill_left == 0:
                    req.is_decode = True
                    new_decodes.append(req)
                else:
                    new_waiting.append(req)
            else:
                new_waiting.append(req)

        self.active_decodes = new_decodes
        self.waiting_prefill = new_waiting
        return ans
