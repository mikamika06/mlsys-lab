class ToyZeRO1Optimizer:

    def __init__(self, params, lr=1e-3, world_size=1, rank=0):
        raise NotImplementedError

    def step(self, grads):
        raise NotImplementedError

    def get_rank_state_bytes(self):
        raise NotImplementedError
