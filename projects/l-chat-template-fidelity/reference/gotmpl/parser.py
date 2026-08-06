from .lexer import Action, TemplateError, Text, tokenize


class Node:
    pass


class TextNode(Node):
    def __init__(self, s):
        self.s = s


class ActionNode(Node):
    def __init__(self, pipe):
        self.pipe = pipe


class IfNode(Node):
    def __init__(self, pipe, then, other):
        self.pipe = pipe
        self.then = then
        self.other = other


class RangeNode(Node):
    def __init__(self, names, pipe, body, other):
        self.names = names
        self.pipe = pipe
        self.body = body
        self.other = other


class WithNode(Node):
    def __init__(self, pipe, body, other):
        self.pipe = pipe
        self.body = body
        self.other = other


class AssignNode(Node):
    def __init__(self, name, pipe, define):
        self.name = name
        self.pipe = pipe
        self.define = define


class Field:
    def __init__(self, base, names):
        self.base = base
        self.names = names


class Var:
    def __init__(self, name):
        self.name = name


class Lit:
    def __init__(self, value):
        self.value = value


class Ident:
    def __init__(self, name):
        self.name = name


class Pipe:
    def __init__(self, cmds):
        self.cmds = cmds


ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'",
           "a": "\a", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def _split_args(s):
    """Top-level whitespace split, honouring quotes and parentheses."""
    out, cur, depth, quote = [], [], 0, None
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'`":
            quote = c
            cur.append(c)
        elif c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise TemplateError("unbalanced ) in %r" % s)
            cur.append(c)
        elif c in " \t\r\n" and depth == 0:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(c)
        i += 1
    if quote:
        raise TemplateError("unterminated string in %r" % s)
    if depth:
        raise TemplateError("unbalanced ( in %r" % s)
    if cur:
        out.append("".join(cur))
    return out


def _split_pipe(s):
    out, cur, depth, quote = [], [], 0, None
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'`":
            quote = c
            cur.append(c)
        elif c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            cur.append(c)
        elif c == "|" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return [x.strip() for x in out]


def _unquote(s):
    if s.startswith('"'):
        if not s.endswith('"') or len(s) < 2:
            raise TemplateError("unterminated string %s" % s)
        body = s[1:-1]
        out, i = [], 0
        while i < len(body):
            if body[i] == "\\" and i + 1 < len(body):
                nxt = body[i + 1]
                if nxt in ESCAPES:
                    out.append(ESCAPES[nxt])
                    i += 2
                    continue
                if nxt == "u" and i + 5 < len(body) + 1:
                    out.append(chr(int(body[i + 2:i + 6], 16)))
                    i += 6
                    continue
            out.append(body[i])
            i += 1
        return "".join(out)
    if s.startswith("`"):
        if not s.endswith("`") or len(s) < 2:
            raise TemplateError("unterminated raw string %s" % s)
        return s[1:-1]
    raise TemplateError("not a string: %s" % s)


def parse_operand(s):
    if not s:
        raise TemplateError("empty operand")
    if s[0] in "\"`":
        return Lit(_unquote(s))
    if s.startswith("("):
        if not s.endswith(")"):
            raise TemplateError("unbalanced parenthesis in %s" % s)
        return parse_pipeline(s[1:-1])
    if s == "true":
        return Lit(True)
    if s == "false":
        return Lit(False)
    if s in ("nil", "null"):
        return Lit(None)
    if s == ".":
        return Field(None, [])
    if s.startswith("$"):
        head, _, rest = s.partition(".")
        names = [x for x in rest.split(".") if x] if rest else []
        return Field(Var(head), names)
    if s.startswith("."):
        return Field(None, [x for x in s[1:].split(".") if x])
    try:
        return Lit(int(s))
    except ValueError:
        pass
    try:
        return Lit(float(s))
    except ValueError:
        pass
    return Ident(s)


def parse_pipeline(s):
    s = s.strip()
    if not s:
        raise TemplateError("empty pipeline")
    cmds = []
    for part in _split_pipe(s):
        args = _split_args(part)
        if not args:
            raise TemplateError("empty command in %r" % s)
        cmds.append([parse_operand(a) for a in args])
    return Pipe(cmds)


def _range_vars(body):
    """`range $i, $m := .Messages` -> (["$i","$m"], ".Messages")."""
    if ":=" not in body:
        return [], body
    left, _, right = body.partition(":=")
    names = [x.strip() for x in left.split(",") if x.strip()]
    for n in names:
        if not n.startswith("$"):
            raise TemplateError("range variable %s must start with $" % n)
    return names, right.strip()


def _is_assign(body):
    head = body.split(None, 1)
    return (len(head) > 1 and head[1].startswith("=")
            and not head[1].startswith("=="))


def parse(src):
    toks = tokenize(src)
    pos = [0]

    def take():
        pos[0] += 1

    def block(stop):
        """Nodes up to the first stop word; the stop token is left unconsumed."""
        out = []
        while pos[0] < len(toks):
            tok = toks[pos[0]]
            if isinstance(tok, Text):
                take()
                if tok.s:
                    out.append(TextNode(tok.s))
                continue
            body = tok.s
            word = body.split(None, 1)[0] if body else ""
            if word in stop:
                return out, word, body[len(word):].strip()
            take()
            if not body:
                continue
            rest = body[len(word):].strip()
            if word == "if":
                out.append(parse_if(rest))
            elif word == "range":
                names, pipe_src = _range_vars(rest)
                body_nodes, closer, _ = block({"else", "end"})
                other = []
                if closer == "else":
                    take()
                    other, closer, _ = block({"end"})
                take()
                out.append(RangeNode(names, parse_pipeline(pipe_src), body_nodes, other))
            elif word == "with":
                body_nodes, closer, _ = block({"else", "end"})
                other = []
                if closer == "else":
                    take()
                    other, closer, _ = block({"end"})
                take()
                out.append(WithNode(parse_pipeline(rest), body_nodes, other))
            elif word in ("end", "else"):
                raise TemplateError("unexpected {{%s}}" % word)
            elif body.startswith("$") and (":=" in body or _is_assign(body)):
                if ":=" in body:
                    name, _, expr = body.partition(":=")
                    define = True
                else:
                    name, _, expr = body.partition("=")
                    define = False
                out.append(AssignNode(name.strip(), parse_pipeline(expr.strip()), define))
            else:
                out.append(ActionNode(parse_pipeline(body)))
        return out, None, ""

    def parse_if(pipe_src):
        """`if` through its matching `end`, including any `else if` chain."""
        then, closer, crest = block({"else", "end"})
        if closer is None:
            raise TemplateError("unclosed {{if}}")
        take()
        if closer == "end":
            return IfNode(parse_pipeline(pipe_src), then, [])
        if crest.startswith("if"):
            return IfNode(parse_pipeline(pipe_src), then, [parse_if(crest[2:].strip())])
        other, closer2, _ = block({"end"})
        if closer2 is None:
            raise TemplateError("unclosed {{if}}")
        take()
        return IfNode(parse_pipeline(pipe_src), then, other)

    nodes, closer, _ = block(set())
    if closer:
        raise TemplateError("unexpected {{%s}} at top level" % closer)
    return nodes
