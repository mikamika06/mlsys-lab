class Router:
    def __init__(self, num_replicas):
        raise NotImplementedError

    def step(self, prompt):
        raise NotImplementedError
