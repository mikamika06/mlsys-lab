import math

_UNARY = {"sin"}
_BINARY = {"add", "sub", "mul"}


def _forward(tape, x):
    """Evaluate every node value; node i (0-indexed) is len(x) + i."""
    n_in = len(x)
    val = list(x) + [0.0] * len(tape)
    for i, (op, ins) in enumerate(tape):
        idx = n_in + i
        if op == "add":
            a, b = ins
            val[idx] = val[a] + val[b]
        elif op == "sub":
            a, b = ins
            val[idx] = val[a] - val[b]
        elif op == "mul":
            a, b = ins
            val[idx] = val[a] * val[b]
        elif op == "sin":
            (a,) = ins
            val[idx] = math.sin(val[a])
        else:
            raise ValueError(f"unknown op {op!r}")
    return val


def tape_grad(tape: list[tuple[str, tuple[int, ...]]], x: list[float]) -> list[float]:
    """
    Reverse-mode autograd over a Wengert list.

    ``tape`` is a list of ``(op, input_node_indices)`` entries, one per
    computed node, already in topological (forward-evaluation) order.
    Node indices ``0 .. len(x)-1`` refer to the entries of ``x``; node
    index ``len(x) + i`` refers to the value produced by ``tape[i]``.
    Supported ops: ``"add"``, ``"sub"``, ``"mul"`` (2 inputs each) and
    ``"sin"`` (1 input). The function's output is the value of the LAST
    node in the tape.

    Returns the gradient of the output with respect to every entry of
    ``x``, as a list of floats.
    """
    n_in = len(x)
    val = _forward(tape, x)
    n_nodes = len(val)

    adj = [0.0] * n_nodes
    adj[-1] = 1.0

    for i in reversed(range(len(tape))):
        op, ins = tape[i]
        idx = n_in + i
        g = adj[idx]
        if op == "add":
            a, b = ins
            adj[a] += g
            adj[b] += g
        elif op == "sub":
            a, b = ins
            adj[a] += g
            adj[b] -= g
        elif op == "mul":
            a, b = ins
            adj[a] += g * val[b]
            adj[b] += g * val[a]
        elif op == "sin":
            (a,) = ins
            adj[a] += g * math.cos(val[a])
        else:
            raise ValueError(f"unknown op {op!r}")

    return adj[:n_in]
