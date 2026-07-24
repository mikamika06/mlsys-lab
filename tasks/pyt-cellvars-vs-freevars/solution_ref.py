import types


def analyze_closure(source: str) -> dict:
    code = compile(source, "<arena>", "exec")
    result = {}

    def walk(obj, prefix):
        if not isinstance(obj, types.CodeType):
            return
        result[prefix] = {
            "co_cellvars": sorted(obj.co_cellvars),
            "co_freevars": sorted(obj.co_freevars),
        }
        for child in obj.co_consts:
            if isinstance(child, types.CodeType):
                child_name = child.co_name if prefix == "module" else prefix + "." + child.co_name
                walk(child, child_name)

    walk(code, "module")
    return result
