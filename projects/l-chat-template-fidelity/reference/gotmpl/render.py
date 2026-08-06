from .eval import Evaluator, Scope
from .lexer import TemplateError
from .parser import parse


def render(src, data, funcs=None):
    nodes = parse(src)
    scope = Scope(None, data, data)
    out = []
    Evaluator(funcs).nodes(nodes, scope, out)
    return "".join(out)
