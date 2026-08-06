import re

from .lexer import TemplateError
from .parser import (ActionNode, AssignNode, Field, Ident, IfNode, Lit, Pipe,
                     RangeNode, TextNode, Var, WithNode)

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def snake(name):
    return _CAMEL.sub("_", name).lower()


def is_true(v):
    if v is None or v is False:
        return False
    if v is True:
        return True
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, (str, bytes, list, tuple, dict, set)):
        return len(v) > 0
    return True


def _field(obj, name):
    if obj is None:
        return None
    key = snake(name)
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        if name in obj:
            return obj[name]
        return None
    return getattr(obj, key, getattr(obj, name, None))


def _num(v):
    if isinstance(v, bool):
        raise TemplateError("cannot compare bool as number")
    if isinstance(v, (int, float)):
        return v
    raise TemplateError("not a number: %r" % (v,))


def _eq(a, b):
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _len(v):
    if v is None:
        return 0
    if isinstance(v, (str, bytes, list, tuple, dict, set)):
        return len(v)
    raise TemplateError("len of untyped %r" % (v,))


def _index(v, *keys):
    for k in keys:
        if v is None:
            return None
        if isinstance(v, dict):
            v = v.get(k)
        else:
            try:
                v = v[k]
            except Exception:
                return None
    return v


BUILTINS = {
    "eq": lambda a, *rest: any(_eq(a, b) for b in rest),
    "ne": lambda a, b: not _eq(a, b),
    "lt": lambda a, b: _num(a) < _num(b),
    "le": lambda a, b: _num(a) <= _num(b),
    "gt": lambda a, b: _num(a) > _num(b),
    "ge": lambda a, b: _num(a) >= _num(b),
    "not": lambda a: not is_true(a),
    "len": _len,
    "index": _index,
}


def _and(args):
    last = True
    for a in args:
        last = a
        if not is_true(a):
            return a
    return last


def _or(args):
    last = False
    for a in args:
        last = a
        if is_true(a):
            return a
    return last


class Scope:
    def __init__(self, parent=None, dot=None, root=None):
        self.parent = parent
        self.vars = {}
        self.dot = dot if parent is None else (dot if dot is not None else parent.dot)
        self.root = root if parent is None else parent.root

    def get(self, name):
        s = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        raise TemplateError("undefined variable %s" % name)

    def define(self, name, value):
        self.vars[name] = value

    def assign(self, name, value):
        s = self
        while s is not None:
            if name in s.vars:
                s.vars[name] = value
                return
            s = s.parent
        raise TemplateError("assignment to undefined variable %s" % name)

    def child(self, dot=None):
        return Scope(self, dot if dot is not None else self.dot)


def to_text(v):
    # A key the data does not carry stands for a zero-valued struct field, and Go
    # prints an empty string for that. `<no value>` is what Go prints for a nil
    # interface, which this data model never produces.
    if v is None:
        return ""
    if v is True:
        return "true"
    if v is False:
        return "false"
    # Go's fmt.Stringer: ollama gives Tools and tool-call Arguments a String
    # method, so printing one whole emits JSON instead of the default formatting.
    if (not isinstance(v, (str, int, float))
            and type(v).__str__ is not object.__str__):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (list, tuple)):
        return "[" + " ".join(to_text(x) for x in v) + "]"
    if isinstance(v, dict):
        return "map[" + " ".join("%s:%s" % (k, to_text(v[k]))
                                 for k in sorted(v)) + "]"
    return str(v)


class Evaluator:
    def __init__(self, funcs=None):
        self.funcs = dict(BUILTINS)
        if funcs:
            self.funcs.update(funcs)

    def operand(self, node, scope):
        if isinstance(node, Lit):
            return node.value
        if isinstance(node, Pipe):
            return self.pipeline(node, scope)
        if isinstance(node, Field):
            if node.base is None:
                base = scope.dot
            elif isinstance(node.base, Var):
                base = scope.root if node.base.name == "$" else scope.get(node.base.name)
            else:
                base = self.operand(node.base, scope)
            for name in node.names:
                base = _field(base, name)
            return base
        if isinstance(node, Ident):
            if node.name not in self.funcs:
                raise TemplateError("function %r not defined" % node.name)
            return self.funcs[node.name]
        raise TemplateError("cannot evaluate %r" % (node,))

    def command(self, args, scope, piped=None):
        head = args[0]
        if isinstance(head, Ident) and head.name in ("and", "or"):
            vals = [self.operand(a, scope) for a in args[1:]]
            if piped is not None:
                vals.append(piped)
            return _and(vals) if head.name == "and" else _or(vals)
        if isinstance(head, Ident):
            fn = self.operand(head, scope)
            vals = [self.operand(a, scope) for a in args[1:]]
            if piped is not None:
                vals.append(piped)
            return fn(*vals)
        if len(args) > 1:
            raise TemplateError("%r is not a function" % (head,))
        v = self.operand(head, scope)
        if piped is not None:
            if not callable(v):
                raise TemplateError("cannot pipe into a non-function")
            return v(piped)
        return v

    def pipeline(self, pipe, scope):
        value = None
        for i, cmd in enumerate(pipe.cmds):
            value = self.command(cmd, scope, None if i == 0 else value)
        return value

    def nodes(self, nodes, scope, out):
        for n in nodes:
            self.node(n, scope, out)

    def node(self, n, scope, out):
        if isinstance(n, TextNode):
            out.append(n.s)
            return
        if isinstance(n, ActionNode):
            out.append(to_text(self.pipeline(n.pipe, scope)))
            return
        if isinstance(n, AssignNode):
            value = self.pipeline(n.pipe, scope)
            if n.define:
                scope.define(n.name, value)
            else:
                scope.assign(n.name, value)
            return
        if isinstance(n, IfNode):
            if is_true(self.pipeline(n.pipe, scope)):
                self.nodes(n.then, scope.child(), out)
            else:
                self.nodes(n.other, scope.child(), out)
            return
        if isinstance(n, WithNode):
            v = self.pipeline(n.pipe, scope)
            if is_true(v):
                self.nodes(n.body, scope.child(v), out)
            else:
                self.nodes(n.other, scope.child(), out)
            return
        if isinstance(n, RangeNode):
            seq = self.pipeline(n.pipe, scope)
            items = self._items(seq)
            if not items:
                self.nodes(n.other, scope.child(), out)
                return
            for key, value in items:
                inner = scope.child(value)
                if len(n.names) == 1:
                    inner.define(n.names[0], value)
                elif len(n.names) >= 2:
                    inner.define(n.names[0], key)
                    inner.define(n.names[1], value)
                self.nodes(n.body, inner, out)
            return
        raise TemplateError("unknown node %r" % (n,))

    @staticmethod
    def _items(seq):
        if seq is None:
            return []
        if isinstance(seq, dict):
            return [(k, seq[k]) for k in sorted(seq)]
        if isinstance(seq, (list, tuple)):
            return list(enumerate(seq))
        if isinstance(seq, str):
            return list(enumerate(seq))
        if isinstance(seq, int):
            return [(i, i) for i in range(seq)]
        raise TemplateError("range over %r" % type(seq).__name__)
