class StatefulRunner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, token):
        raise NotImplementedError
