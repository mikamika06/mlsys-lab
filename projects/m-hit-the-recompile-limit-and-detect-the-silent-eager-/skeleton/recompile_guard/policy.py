class GuardedFunction:
    def __init__(self, fn, recompile_limit=8):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
