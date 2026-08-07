class OptimizationProfile:
    def __init__(self, name="default"):
        raise NotImplementedError

    def add_shape(self, name, min_shape, opt_shape, max_shape):
        raise NotImplementedError

    def validate_shape(self, name, shape):
        raise NotImplementedError

    def profile_hash(self):
        raise NotImplementedError
