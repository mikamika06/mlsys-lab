class Primitive:
    def __init__(self, name: str):
        raise NotImplementedError


class JaxprEqn:
    def __init__(self, primitive, invars=None, outvars=None, params=None):
        raise NotImplementedError


class Jaxpr:
    def __init__(self, eqns=None, invars=None, outvars=None):
        raise NotImplementedError


def count_equations(jaxpr):
    raise NotImplementedError


def list_unique_primitives(jaxpr):
    raise NotImplementedError


def analyze_jaxpr(jaxpr):
    raise NotImplementedError
