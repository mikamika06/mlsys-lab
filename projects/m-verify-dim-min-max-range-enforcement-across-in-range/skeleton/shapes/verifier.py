class DimType:
    AUTO = "AUTO"
    DYNAMIC = "DYNAMIC"
    EXPLICIT = "EXPLICIT"


class ConstraintViolationError(Exception):
    def __init__(self, name, val, min_val, max_val):
        raise NotImplementedError

    def suggested_fix(self):
        raise NotImplementedError


class Dim:
    def __init__(self, name, min_val=1, max_val=2147483647, dim_type=DimType.EXPLICIT):
        raise NotImplementedError

    @classmethod
    def auto(cls, name):
        raise NotImplementedError

    @classmethod
    def dynamic(cls, name):
        raise NotImplementedError


def verify_range(dim, batch_size):
    raise NotImplementedError


def resolve_module_signature(dim_specs, observed_shapes):
    raise NotImplementedError
