class Parameter:

    def __init__(self, data):
        raise NotImplementedError


class DummyOptimizer:

    def __init__(self, param_groups):
        raise NotImplementedError


class GradScaler:

    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        raise NotImplementedError

    def get_scale(self):
        raise NotImplementedError

    def unscale_(self, optimizer):
        raise NotImplementedError
