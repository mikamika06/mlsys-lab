class ConstraintViolationError(ValueError):
    def __init__(self, message, suggested_fix=None):
        super().__init__(message)
        self.suggested_fix = suggested_fix


def verify_range(module, sample_inputs, min_val, max_val):
    raise NotImplementedError


def execute_with_constraints(module, inputs, dim_spec):
    raise NotImplementedError
