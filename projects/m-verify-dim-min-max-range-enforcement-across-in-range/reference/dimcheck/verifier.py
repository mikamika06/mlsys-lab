class ConstraintViolationError(ValueError):
    def __init__(self, message, suggested_fix=None):
        super().__init__(message)
        self.suggested_fix = suggested_fix


def verify_range(module, sample_inputs, min_val, max_val):
    res = []
    for x in sample_inputs:
        if x < min_val or x > max_val:
            raise ConstraintViolationError(
                f"Value {x} out of range [{min_val}, {max_val}]",
                suggested_fix=f"Adjust input or set dim range to encompass {x}"
            )
        res.append(module(x))
    return res


def execute_with_constraints(module, inputs, dim_spec):
    min_v, max_v = dim_spec
    return verify_range(module, inputs, min_v, max_v)
