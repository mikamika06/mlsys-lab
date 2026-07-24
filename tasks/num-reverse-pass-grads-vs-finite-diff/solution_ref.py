import math


def backward_pass(tape, values, n_inputs):
    """Run reverse-mode autodiff over a Wengert tape.

    `tape` is a list of `(op, args)` pairs. `values` is the full forward
    trace: `values[0:n_inputs]` are the leaf inputs and `values[n_inputs+k]`
    is the value produced by `tape[k]`, computed from earlier entries only
    (so the tape is already in topological order). `values[-1]` is the
    scalar output.

    Returns a list/array of length `n_inputs`: d(values[-1]) / d(values[i])
    for each input i, obtained by walking the tape in reverse and
    accumulating local-derivative contributions into every argument a node
    reads from (a node used more than once must have its contributions
    summed, not overwritten).
    """
    n_total = len(values)
    grad = [0.0] * n_total
    grad[n_total - 1] = 1.0  # seed: d(output)/d(output) = 1

    for k in range(len(tape) - 1, -1, -1):
        op, args = tape[k]
        node_idx = n_inputs + k
        g = grad[node_idx]
        if op == "add":
            a, b = args
            grad[a] += g
            grad[b] += g
        elif op == "mul":
            a, b = args
            grad[a] += g * values[b]
            grad[b] += g * values[a]
        elif op == "sin":
            (a,) = args
            grad[a] += g * math.cos(values[a])
        elif op == "exp":
            (a,) = args
            grad[a] += g * values[node_idx]  # d/da exp(a) = exp(a) = values[node_idx]
        else:
            raise ValueError(f"unknown op {op!r}")

    return grad[:n_inputs]
