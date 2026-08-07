class MoEModelSimulator:
    def __init__(self, num_experts: int, capacity_factor: float):
        raise NotImplementedError

    def forward(self, tokens, drop_tokens=True):
        raise NotImplementedError
