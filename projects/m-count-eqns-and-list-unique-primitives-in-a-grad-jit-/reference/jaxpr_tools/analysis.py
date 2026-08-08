class Primitive:
    def __init__(self, name: str):
        self.name = name


class JaxprEqn:
    def __init__(self, primitive: Primitive, invars=None, outvars=None, params=None):
        self.primitive = primitive
        self.invars = invars if invars is not None else []
        self.outvars = outvars if outvars is not None else []
        self.params = params if params is not None else {}


class Jaxpr:
    def __init__(self, eqns=None, invars=None, outvars=None):
        self.eqns = eqns if eqns is not None else []
        self.invars = invars if invars is not None else []
        self.outvars = outvars if outvars is not None else []


def _extract_jaxprs(val):
    found = []
    if hasattr(val, "eqns") and isinstance(getattr(val, "eqns", None), list):
        found.append(val)
    elif isinstance(val, (list, tuple)):
        for item in val:
            found.extend(_extract_jaxprs(item))
    elif isinstance(val, dict):
        for item in val.values():
            found.extend(_extract_jaxprs(item))
    return found


def count_equations(jaxpr: Jaxpr) -> int:
    total = len(jaxpr.eqns)
    for eqn in jaxpr.eqns:
        for p_val in eqn.params.values():
            for sub in _extract_jaxprs(p_val):
                total += count_equations(sub)
    return total


def list_unique_primitives(jaxpr: Jaxpr) -> list:
    names = set()
    for eqn in jaxpr.eqns:
        names.add(eqn.primitive.name)
        for p_val in eqn.params.values():
            for sub in _extract_jaxprs(p_val):
                names.update(list_unique_primitives(sub))
    return sorted(list(names))


def analyze_jaxpr(jaxpr: Jaxpr) -> dict:
    return {
        "eqn_count": count_equations(jaxpr),
        "unique_primitives": list_unique_primitives(jaxpr),
    }
