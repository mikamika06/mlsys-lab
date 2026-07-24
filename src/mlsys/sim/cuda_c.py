"""A tiny interpreter for a restricted subset of real CUDA-C.

The learner writes an actual `.cu` kernel (real syntax: `__global__`, pointer
params, `threadIdx.x`, `__shared__`, `__syncthreads()`, control flow, ...).
`CudaProgram` parses it into an AST and `launch()` executes it thread-by-thread
on the existing software GPU in `mlsys.sim`, so every `a[i]` read/write
goes through `Thread.gload`/`gstore` (global memory) or `sload`/`sstore`
(`__shared__` memory) and gets measured exactly like a real access pattern —
coalescing, bank conflicts, cycles, all of it come from `sim.gpu.GPU.launch`.

Supported syntax:
  __global__ void name(float* a, const float* b, int n, float s) { ... }
  int i = expr; float x;                 (int/float/unsigned/size_t/long/... -> plain numbers)
  x = e;  x += e; -= *= /=;  x++; x--;
  threadIdx.x/.y  blockIdx.x/.y  blockDim.x/.y  gridDim.x/.y   (this sim is 1D: .y/.z read as 0 or 1)
  a[i]                                    (pointer params -> gmem, __shared__ arrays -> smem)
  __shared__ float s[64];
  __syncthreads();
  + - * / %   - !   < <= > >= == !=   && ||   ?:   ( )
  if/else, for(;;), while, { }, return;
  min max fminf fmaxf sqrtf expf logf fabsf floorf ceilf powf
  __shfl_up_sync/__shfl_down_sync/__shfl_xor_sync(mask, var, delta)  -- warp shuffle;
      only as a whole right-hand side: `v = __shfl_up_sync(0xffffffff, v, 1);`
      or an initializer `float up = __shfl_up_sync(...);` (it is warp-synchronous,
      so it must reach GPU.launch the same way a barrier does)

Not supported (raises a ValueError at parse time): structs, multi-dim arrays,
function calls other than the math builtins above, pointer arithmetic other
than `name[expr]`, multiple kernels sharing the same name, `break`/`continue`,
`switch`, preprocessor directives.
"""
from __future__ import annotations

import math
import re

from .gpu import ShflRequest

__all__ = ["CudaProgram", "CudaParseError"]


class CudaParseError(ValueError):
    """Raised for any malformed CUDA-C source. Always a ValueError."""


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("COMMENT_BLOCK", r"/\*.*?\*/"),
    ("COMMENT_LINE", r"//[^\n]*"),
    ("NUMBER", r"0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|[-+*/%=<>!?:;,(){}\[\]\.]"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("MISMATCH", r"."),
]
_MASTER_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC), re.DOTALL)


class _Tok:
    __slots__ = ("type", "value", "line")

    def __init__(self, type_, value, line):
        self.type = type_
        self.value = value
        self.line = line

    def __repr__(self):
        return f"{self.type}:{self.value!r}@L{self.line}"


def _parse_number(text: str):
    if text[:2].lower() == "0x":          # warp masks are written 0xffffffff
        return int(text.rstrip("uUlL"), 16)
    i = len(text)
    while i > 0 and text[i - 1] in "fFuUlL":
        i -= 1
    core, suf = text[:i], text[i:].lower()
    if "." in core or "e" in core.lower() or "f" in suf:
        return float(core)
    return int(core)


def tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    line = 1
    pos = 0
    n = len(src)
    while pos < n:
        m = _MASTER_RE.match(src, pos)
        if m is None:  # pragma: no cover — MISMATCH's '.' always matches
            raise CudaParseError(f"line {line}: cannot tokenize near {src[pos:pos + 20]!r}")
        kind = m.lastgroup
        val = m.group()
        pos = m.end()
        if kind == "NEWLINE":
            line += 1
            continue
        if kind in ("SKIP", "COMMENT_LINE"):
            continue
        if kind == "COMMENT_BLOCK":
            line += val.count("\n")
            continue
        if kind == "MISMATCH":
            raise CudaParseError(f"line {line}: unexpected character {val!r}")
        toks.append(_Tok(kind, val, line))
    toks.append(_Tok("EOF", "", line))
    return toks


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------

class Num:
    def __init__(self, value):
        self.value = value


class Var:
    def __init__(self, name):
        self.name = name


class Member:
    def __init__(self, obj, field):
        self.obj = obj
        self.field = field


class Index:
    def __init__(self, arr, idx):
        self.arr = arr
        self.idx = idx


class Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Unary:
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr


class PreIncDec:
    def __init__(self, op, target):
        self.op = op
        self.target = target


class PostIncDec:
    def __init__(self, op, target):
        self.op = op
        self.target = target


class Binary:
    def __init__(self, op, l, r):
        self.op = op
        self.l = l
        self.r = r


class Logical:
    def __init__(self, op, l, r):
        self.op = op
        self.l = l
        self.r = r


class Ternary:
    def __init__(self, cond, t, f):
        self.cond = cond
        self.t = t
        self.f = f


class Assign:
    def __init__(self, op, target, value):
        self.op = op
        self.target = target
        self.value = value


class VarDecl:
    def __init__(self, type_, name, init):
        self.type = type_
        self.name = name
        self.init = init


class SharedDecl:
    def __init__(self, type_, name, size):
        self.type = type_
        self.name = name
        self.size = size


class ExprStmt:
    def __init__(self, expr):
        self.expr = expr


class If:
    def __init__(self, cond, then, els):
        self.cond = cond
        self.then = then
        self.els = els


class For:
    def __init__(self, init, cond, step, body):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body


class While:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body


class Block:
    def __init__(self, stmts):
        self.stmts = stmts


class Return:
    def __init__(self, expr):
        self.expr = expr


class Sync:
    pass


class Param:
    def __init__(self, name, is_ptr, is_const):
        self.name = name
        self.is_ptr = is_ptr
        self.is_const = is_const


class Function:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body
        self.shared_layout: dict[str, tuple[int, int]] = {}


TYPE_WORDS = {"void", "int", "float", "double", "unsigned", "long", "short", "char", "size_t", "bool"}
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/="}
ARITH_OPS = {"+", "-", "*", "/", "%"}


# --------------------------------------------------------------------------
# Parser (recursive descent)
# --------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[_Tok]):
        self.toks = tokens
        self.pos = 0

    def peek(self, k=0):
        i = self.pos + k
        return self.toks[i] if i < len(self.toks) else self.toks[-1]

    def at(self, value=None, type_=None) -> bool:
        t = self.peek()
        if value is not None and t.value != value:
            return False
        if type_ is not None and t.type != type_:
            return False
        return True

    def advance(self) -> _Tok:
        t = self.toks[self.pos]
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def expect(self, value=None, type_=None) -> _Tok:
        t = self.peek()
        if (value is not None and t.value != value) or (type_ is not None and t.type != type_):
            want = value if value is not None else type_
            raise CudaParseError(f"line {t.line}: expected {want!r} but found {t.value!r}")
        return self.advance()

    # -- top level --------------------------------------------------------

    def parse_program(self) -> dict[str, Function]:
        funcs: dict[str, Function] = {}
        while not self.at(type_="EOF"):
            fn = self.parse_function()
            funcs[fn.name] = fn
        return funcs

    def parse_type(self):
        is_const = False
        if self.at(value="const"):
            is_const = True
            self.advance()
        words = []
        while self.at(type_="ID") and self.peek().value in TYPE_WORDS:
            words.append(self.advance().value)
        # trailing 'const', e.g. "float const"
        if self.at(value="const"):
            is_const = True
            self.advance()
        if not words:
            t = self.peek()
            raise CudaParseError(f"line {t.line}: expected a type (int/float/...) but found {t.value!r}")
        return " ".join(words), is_const

    def parse_function(self) -> Function:
        if not self.at(value="__global__"):
            t = self.peek()
            raise CudaParseError(f"line {t.line}: expected '__global__' kernel declaration, found {t.value!r}")
        self.advance()
        self.parse_type()  # return type, must be void — not enforced beyond parsing
        name = self.expect(type_="ID").value
        self.expect(value="(")
        params = []
        if not self.at(value=")"):
            while True:
                params.append(self.parse_param())
                if self.at(value=","):
                    self.advance()
                    continue
                break
        self.expect(value=")")
        body = self.parse_block()
        fn = Function(name, params, body)
        fn.shared_layout = _collect_shared_layout(body)
        return fn

    def parse_param(self) -> Param:
        _, is_const = self.parse_type()
        is_ptr = False
        if self.at(value="*"):
            is_ptr = True
            self.advance()
        name = self.expect(type_="ID").value
        return Param(name, is_ptr, is_const)

    # -- statements ---------------------------------------------------------

    def parse_block(self) -> Block:
        self.expect(value="{")
        stmts = []
        while not self.at(value="}"):
            if self.at(type_="EOF"):
                raise CudaParseError("unexpected end of source: unclosed '{'")
            stmts.append(self.parse_statement())
        self.expect(value="}")
        return Block(stmts)

    def parse_statement(self):
        t = self.peek()
        if t.value == "{":
            return self.parse_block()
        if t.value == "if":
            return self.parse_if()
        if t.value == "for":
            return self.parse_for()
        if t.value == "while":
            return self.parse_while()
        if t.value == "return":
            self.advance()
            expr = None if self.at(value=";") else self.parse_expr()
            self.expect(value=";")
            return Return(expr)
        if t.value == "__syncthreads":
            self.advance()
            self.expect(value="(")
            self.expect(value=")")
            self.expect(value=";")
            return Sync()
        if t.value == "__shared__":
            return self.parse_shared_decl()
        if (t.type == "ID" and t.value in TYPE_WORDS) or t.value == "const":
            return self.parse_var_decl()
        expr = self.parse_expr()
        self.expect(value=";")
        return ExprStmt(expr)

    def parse_var_decl(self) -> VarDecl:
        self.parse_type()
        name = self.expect(type_="ID").value
        init = None
        if self.at(value="="):
            self.advance()
            init = self.parse_assignment()
        self.expect(value=";")
        return VarDecl(None, name, init)

    def parse_shared_decl(self) -> SharedDecl:
        self.expect(value="__shared__")
        self.parse_type()
        name = self.expect(type_="ID").value
        self.expect(value="[")
        size_expr = self.parse_expr()
        self.expect(value="]")
        self.expect(value=";")
        return SharedDecl(None, name, size_expr)

    def parse_if(self) -> If:
        self.expect(value="if")
        self.expect(value="(")
        cond = self.parse_expr()
        self.expect(value=")")
        then = self.parse_statement()
        els = None
        if self.at(value="else"):
            self.advance()
            els = self.parse_statement()
        return If(cond, then, els)

    def parse_while(self) -> While:
        self.expect(value="while")
        self.expect(value="(")
        cond = self.parse_expr()
        self.expect(value=")")
        return While(cond, self.parse_statement())

    def parse_for(self) -> For:
        self.expect(value="for")
        self.expect(value="(")
        init = None
        if not self.at(value=";"):
            t = self.peek()
            if (t.type == "ID" and t.value in TYPE_WORDS) or t.value == "const":
                self.parse_type()
                name = self.expect(type_="ID").value
                iv = None
                if self.at(value="="):
                    self.advance()
                    iv = self.parse_assignment()
                init = VarDecl(None, name, iv)
            else:
                init = ExprStmt(self.parse_expr())
        self.expect(value=";")
        cond = None if self.at(value=";") else self.parse_expr()
        self.expect(value=";")
        step = None if self.at(value=")") else self.parse_expr()
        self.expect(value=")")
        body = self.parse_statement()
        return For(init, cond, step, body)

    # -- expressions (precedence climbing) ---------------------------------

    def parse_expr(self):
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_ternary()
        if self.peek().value in ASSIGN_OPS:
            op = self.advance().value
            right = self.parse_assignment()
            if not isinstance(left, (Var, Index)):
                raise CudaParseError("invalid assignment target (must be a variable or array element)")
            return Assign(op, left, right)
        return left

    def parse_ternary(self):
        cond = self.parse_logical_or()
        if self.at(value="?"):
            self.advance()
            t = self.parse_assignment()
            self.expect(value=":")
            f = self.parse_ternary()
            return Ternary(cond, t, f)
        return cond

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self.at(value="||"):
            self.advance()
            left = Logical("||", left, self.parse_logical_and())
        return left

    def parse_logical_and(self):
        left = self.parse_equality()
        while self.at(value="&&"):
            self.advance()
            left = Logical("&&", left, self.parse_equality())
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while self.peek().value in ("==", "!="):
            op = self.advance().value
            left = Binary(op, left, self.parse_relational())
        return left

    def parse_relational(self):
        left = self.parse_additive()
        while self.peek().value in ("<", "<=", ">", ">="):
            op = self.advance().value
            left = Binary(op, left, self.parse_additive())
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek().value in ("+", "-"):
            op = self.advance().value
            left = Binary(op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.peek().value in ("*", "/", "%"):
            op = self.advance().value
            left = Binary(op, left, self.parse_unary())
        return left

    def parse_unary(self):
        t = self.peek()
        if t.value in ("-", "!"):
            self.advance()
            return Unary(t.value, self.parse_unary())
        if t.value in ("++", "--"):
            self.advance()
            return PreIncDec(t.value, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            t = self.peek()
            if t.value == "[":
                self.advance()
                idx = self.parse_expr()
                self.expect(value="]")
                expr = Index(expr, idx)
            elif t.value == "(":
                self.advance()
                args = []
                if not self.at(value=")"):
                    while True:
                        args.append(self.parse_assignment())
                        if self.at(value=","):
                            self.advance()
                            continue
                        break
                self.expect(value=")")
                if not isinstance(expr, Var):
                    raise CudaParseError(f"line {t.line}: call target must be a plain function name")
                expr = Call(expr.name, args)
            elif t.value == ".":
                self.advance()
                field = self.expect(type_="ID").value
                expr = Member(expr, field)
            elif t.value in ("++", "--"):
                self.advance()
                expr = PostIncDec(t.value, expr)
            else:
                break
        return expr

    def parse_primary(self):
        t = self.peek()
        if t.type == "NUMBER":
            self.advance()
            return Num(_parse_number(t.value))
        if t.value == "(":
            self.advance()
            e = self.parse_expr()
            self.expect(value=")")
            return e
        if t.type == "ID":
            self.advance()
            return Var(t.value)
        raise CudaParseError(f"line {t.line}: unexpected token {t.value!r}")


# -- shared-memory static layout -------------------------------------------

def _const_eval(expr):
    if isinstance(expr, Num):
        return expr.value
    if isinstance(expr, Unary) and expr.op == "-":
        return -_const_eval(expr.expr)
    if isinstance(expr, Binary):
        l, r = _const_eval(expr.l), _const_eval(expr.r)
        if expr.op == "+":
            return l + r
        if expr.op == "-":
            return l - r
        if expr.op == "*":
            return l * r
        if expr.op == "/":
            return l // r if isinstance(l, int) and isinstance(r, int) else l / r
    raise CudaParseError("__shared__ array size must be a compile-time constant")


def _walk_shared(stmt, out):
    if stmt is None:
        return
    if isinstance(stmt, Block):
        for s in stmt.stmts:
            _walk_shared(s, out)
    elif isinstance(stmt, If):
        _walk_shared(stmt.then, out)
        _walk_shared(stmt.els, out)
    elif isinstance(stmt, For):
        _walk_shared(stmt.init, out)
        _walk_shared(stmt.body, out)
    elif isinstance(stmt, While):
        _walk_shared(stmt.body, out)
    elif isinstance(stmt, SharedDecl):
        out.append(stmt)


def _collect_shared_layout(body: Block) -> dict[str, tuple[int, int]]:
    decls: list[SharedDecl] = []
    _walk_shared(body, decls)
    layout: dict[str, tuple[int, int]] = {}
    base = 0
    for d in decls:
        size = int(_const_eval(d.size))
        if size <= 0:
            raise CudaParseError(f"__shared__ array '{d.name}' must have a positive size")
        layout[d.name] = (base, size)
        base += size
    return layout


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------

def _truthy(v) -> bool:
    return v != 0


class _Ptr:
    __slots__ = ("base",)

    def __init__(self, base):
        self.base = base


class _Shared:
    __slots__ = ("base", "size")

    def __init__(self, base, size):
        self.base = base
        self.size = size


class _ReturnSignal(Exception):
    pass


_DIM_FIELDS = {"x", "y", "z"}
_MATH_1ARG = {
    "sqrtf": math.sqrt, "expf": math.exp, "logf": math.log,
    "fabsf": abs, "floorf": math.floor, "ceilf": math.ceil,
}
_MATH_2ARG = {
    "min": min, "max": max, "fminf": min, "fmaxf": max, "powf": math.pow,
}


_SHFL = {"__shfl_up_sync": "up", "__shfl_down_sync": "down", "__shfl_xor_sync": "xor",
         "__shfl_up": "up", "__shfl_down": "down", "__shfl_xor": "xor"}


def _is_shfl(node):
    return type(node) is Call and node.name in _SHFL


class Interpreter:
    def __init__(self, fn: Function, params: dict):
        self.fn = fn
        self.params = params

    def run(self, t):
        """Generator: runs the kernel body for thread `t`, yielding once at
        every `__syncthreads()`. `GPU.launch` drives this (and its siblings in
        the same block) round-robin so a barrier really does release only
        once every thread in the block has reached it — see sim.gpu.GPU.launch.
        """
        env: dict = {}
        for p in self.fn.params:
            if p.name not in self.params:
                raise ValueError(f"kernel '{self.fn.name}': missing argument for param '{p.name}'")
            v = self.params[p.name]
            env[p.name] = _Ptr(int(v)) if p.is_ptr else v
        try:
            yield from self._exec_block(self.fn.body, env, t)
        except _ReturnSignal:
            pass

    # statements
    #
    # These are all generator functions (they contain a `yield` — directly in
    # the Sync branch, or transitively via `yield from` into nested
    # statements) so a `__syncthreads()` anywhere inside pauses execution all
    # the way up to `run()`/`GPU.launch`, no matter how deeply it's nested in
    # blocks/if/for/while. Branches that never reach the `yield` behave
    # exactly like an ordinary (non-pausing) function call.

    def _exec_block(self, block: Block, env, t):
        for stmt in block.stmts:
            yield from self._exec_stmt(stmt, env, t)

    def _exec_stmt(self, stmt, env, t):
        k = type(stmt)
        if k is Block:
            yield from self._exec_block(stmt, env, t)
        elif k is VarDecl:
            if _is_shfl(stmt.init):
                env[stmt.name] = yield from self._shfl(stmt.init, env, t)
            else:
                env[stmt.name] = self._eval(stmt.init, env, t) if stmt.init is not None else 0
        elif k is SharedDecl:
            base, size = self.fn.shared_layout[stmt.name]
            env[stmt.name] = _Shared(base, size)
        elif k is ExprStmt:
            # `v = __shfl_up_sync(...)` / `v += __shfl_down_sync(...)` — a shuffle
            # is warp-synchronous, so it has to travel up to GPU.launch the same
            # way a barrier does. Expressions cannot yield in this interpreter, so
            # a shuffle is only legal as the whole right-hand side of an
            # assignment or an initializer (which is how real kernels write it).
            e = stmt.expr
            if type(e) is Assign and _is_shfl(e.value):
                got = yield from self._shfl(e.value, env, t)
                if e.op != "=":
                    got = self._binop(e.op[0], self._eval(e.target, env, t), got)
                    t.alu(1)
                self._store(e.target, got, env, t)
            else:
                self._eval(stmt.expr, env, t)
        elif k is If:
            if _truthy(self._eval(stmt.cond, env, t)):
                yield from self._exec_stmt(stmt.then, env, t)
            elif stmt.els is not None:
                yield from self._exec_stmt(stmt.els, env, t)
        elif k is For:
            if stmt.init is not None:
                yield from self._exec_stmt(stmt.init, env, t)
            while stmt.cond is None or _truthy(self._eval(stmt.cond, env, t)):
                yield from self._exec_stmt(stmt.body, env, t)
                if stmt.step is not None:
                    self._eval(stmt.step, env, t)
        elif k is While:
            while _truthy(self._eval(stmt.cond, env, t)):
                yield from self._exec_stmt(stmt.body, env, t)
        elif k is Return:
            if stmt.expr is not None:
                self._eval(stmt.expr, env, t)
            raise _ReturnSignal()
        elif k is Sync:
            yield  # release control back up to GPU.launch's per-block barrier scheduler
        else:  # pragma: no cover
            raise ValueError(f"cannot execute statement of type {k.__name__}")

    # expressions

    def _eval(self, node, env, t):
        k = type(node)
        if k is Num:
            return node.value
        if k is Var:
            if node.name not in env:
                raise ValueError(f"undefined variable '{node.name}'")
            v = env[node.name]
            if isinstance(v, (_Ptr, _Shared)):
                raise ValueError(f"'{node.name}' is an array/pointer; index it like {node.name}[i]")
            return v
        if k is Member:
            return self._dim(node, t)
        if k is Index:
            return self._load(node, env, t)
        if k is Unary:
            v = self._eval(node.expr, env, t)
            return -v if node.op == "-" else (0 if _truthy(v) else 1)
        if k is Binary:
            l = self._eval(node.l, env, t)
            r = self._eval(node.r, env, t)
            if node.op in ARITH_OPS:
                t.alu(1)
            return self._binop(node.op, l, r)
        if k is Logical:
            l = self._eval(node.l, env, t)
            if node.op == "&&":
                return 1 if (_truthy(l) and _truthy(self._eval(node.r, env, t))) else 0
            return 1 if (_truthy(l) or _truthy(self._eval(node.r, env, t))) else 0
        if k is Ternary:
            branch = node.t if _truthy(self._eval(node.cond, env, t)) else node.f
            return self._eval(branch, env, t)
        if k is Call:
            return self._call(node, env, t)
        if k is Assign:
            return self._assign(node, env, t)
        if k is PreIncDec:
            old = self._eval(node.target, env, t)
            new = old + (1 if node.op == "++" else -1)
            self._store(node.target, new, env, t)
            return new
        if k is PostIncDec:
            old = self._eval(node.target, env, t)
            new = old + (1 if node.op == "++" else -1)
            self._store(node.target, new, env, t)
            return old
        raise ValueError(f"cannot evaluate expression of type {k.__name__}")  # pragma: no cover

    def _dim(self, node: Member, t):
        if not isinstance(node.obj, Var) or node.obj.name not in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
            raise ValueError("unsupported member access (only threadIdx/blockIdx/blockDim/gridDim.x/.y/.z)")
        if node.field not in _DIM_FIELDS:
            raise ValueError(f"unsupported dimension field '.{node.field}'")
        name = node.obj.name
        # this simulator models a 1D grid/block: '.x' carries the real value,
        # '.y'/'.z' read as the identity element (0 for index-like, 1 for size-like)
        if name == "threadIdx":
            return t.threadIdx if node.field == "x" else 0
        if name == "blockIdx":
            return t.blockIdx if node.field == "x" else 0
        if name == "blockDim":
            return t.blockDim if node.field == "x" else 1
        return t.gridDim if node.field == "x" else 1  # gridDim

    def _load(self, node: Index, env, t):
        if not isinstance(node.arr, Var):
            raise ValueError("only simple array indexing name[expr] is supported")
        name = node.arr.name
        container = env.get(name)
        idx = int(self._eval(node.idx, env, t))
        if isinstance(container, _Ptr):
            return t.gload(container.base + idx)
        if isinstance(container, _Shared):
            return t.sload(container.base + idx)
        raise ValueError(f"'{name}' is not a pointer parameter or __shared__ array")

    def _store(self, target, value, env, t):
        if isinstance(target, Var):
            if target.name not in env or isinstance(env[target.name], (_Ptr, _Shared)):
                raise ValueError(f"cannot assign to '{target.name}'")
            env[target.name] = value
            return value
        if isinstance(target, Index):
            if not isinstance(target.arr, Var):
                raise ValueError("only simple array indexing name[expr] is supported")
            name = target.arr.name
            container = env.get(name)
            idx = int(self._eval(target.idx, env, t))
            if isinstance(container, _Ptr):
                t.gstore(container.base + idx, value)
                return value
            if isinstance(container, _Shared):
                t.sstore(container.base + idx, value)
                return value
            raise ValueError(f"'{name}' is not a pointer parameter or __shared__ array")
        raise ValueError("invalid assignment target")  # pragma: no cover — parser already filters this

    def _assign(self, node: Assign, env, t):
        if node.op == "=":
            return self._store(node.target, self._eval(node.value, env, t), env, t)
        old = self._eval(node.target, env, t)
        rhs = self._eval(node.value, env, t)
        t.alu(1)
        new = self._binop(node.op[0], old, rhs)
        return self._store(node.target, new, env, t)

    def _call(self, node: Call, env, t):
        args = [self._eval(a, env, t) for a in node.args]
        if node.name == "__syncthreads":
            return 0
        if node.name in _MATH_1ARG:
            if len(args) != 1:
                raise ValueError(f"{node.name}() takes 1 argument")
            return _MATH_1ARG[node.name](args[0])
        if node.name in _MATH_2ARG:
            if len(args) != 2:
                raise ValueError(f"{node.name}() takes 2 arguments")
            return _MATH_2ARG[node.name](args[0], args[1])
        raise ValueError(f"unsupported function call '{node.name}(...)'")

    def _shfl(self, node: Call, env, t):
        """Execute one warp shuffle: `__shfl_up_sync(mask, var, delta)`.

        Yields a ShflRequest to GPU.launch, which collects the value every lane
        published at this same program point and sends each lane back the value
        of the lane it asked for — real warp-synchronous semantics, the same
        machinery that makes `__syncthreads()` work here.
        """
        kind = _SHFL[node.name]
        args = [self._eval(a, env, t) for a in node.args]
        if node.name.endswith("_sync"):
            if len(args) < 3:
                raise ValueError(f"{node.name}(mask, var, delta) takes 3 arguments")
            val, delta = args[1], args[2]
        else:
            if len(args) < 2:
                raise ValueError(f"{node.name}(var, delta) takes 2 arguments")
            val, delta = args[0], args[1]
        got = yield ShflRequest(kind, val, int(delta))
        return got if got is not None else val

    @staticmethod
    def _binop(op, l, r):
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            if isinstance(l, int) and isinstance(r, int):
                if r == 0:
                    raise ValueError("integer division by zero")
                q = abs(l) // abs(r)
                return q if (l < 0) == (r < 0) else -q
            return l / r
        if op == "%":
            if isinstance(l, int) and isinstance(r, int):
                if r == 0:
                    raise ValueError("integer modulo by zero")
                m = abs(l) % abs(r)
                return m if l >= 0 else -m
            return math.fmod(l, r)
        if op == "<":
            return 1 if l < r else 0
        if op == "<=":
            return 1 if l <= r else 0
        if op == ">":
            return 1 if l > r else 0
        if op == ">=":
            return 1 if l >= r else 0
        if op == "==":
            return 1 if l == r else 0
        if op == "!=":
            return 1 if l != r else 0
        raise ValueError(f"unknown operator {op!r}")  # pragma: no cover


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

class CudaProgram:
    """Parses a restricted CUDA-C source string; launches kernels on a
    `mlsys.sim.GPU` and returns its metrics dict.

        prog = CudaProgram(src_text)
        metrics = prog.launch(gpu, "kernelName", grid, block, params)
    """

    def __init__(self, src_text: str):
        try:
            toks = tokenize(src_text)
            self.funcs: dict[str, Function] = Parser(toks).parse_program()
        except CudaParseError:
            raise
        except Exception as e:  # any other parser bug still surfaces as a clear ValueError
            raise ValueError(f"CUDA-C parse error: {e}") from e
        if not self.funcs:
            raise ValueError("CUDA-C parse error: no __global__ kernel found in source")

    def launch(self, gpu, kernel_name: str, grid: int, block: int, params: dict):
        """params: dict of param name -> int base offset (pointers) or number (scalars)."""
        fn = self.funcs.get(kernel_name)
        if fn is None:
            raise ValueError(f"no such kernel '{kernel_name}'; available: {sorted(self.funcs)}")
        interp = Interpreter(fn, params)

        def _kernel(thread):
            # a generator function (contains `yield from`) so
            # `GPU.launch` detects it and drives it with real barrier
            # semantics at every __syncthreads() — see sim.gpu.GPU.launch.
            yield from interp.run(thread)

        return gpu.launch(_kernel, grid, block)
