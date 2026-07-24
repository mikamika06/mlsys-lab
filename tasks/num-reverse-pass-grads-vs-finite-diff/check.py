import math
import numpy as np


def _forward(tape, inputs):
    """Replay the tape forward: vals[0:n_inputs] = inputs, then one value
    per tape entry, each built only from earlier entries."""
    vals = list(inputs)
    for op, args in tape:
        if op == "add":
            vals.append(vals[args[0]] + vals[args[1]])
        elif op == "mul":
            vals.append(vals[args[0]] * vals[args[1]])
        elif op == "sin":
            vals.append(math.sin(vals[args[0]]))
        elif op == "exp":
            vals.append(math.exp(vals[args[0]]))
        else:
            raise ValueError(f"unknown op {op!r}")
    return vals


def _numeric_grad(tape, inputs, h=1e-5):
    """Central-difference oracle: no analytic derivatives, just re-runs the
    forward pass with each input perturbed by +-h."""
    n = len(inputs)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xp = list(inputs); xp[i] += h
        xm = list(inputs); xm[i] -= h
        fp = _forward(tape, xp)[-1]
        fm = _forward(tape, xm)[-1]
        grad[i] = (fp - fm) / (2.0 * h)
    return grad


# Each entry: (n_inputs, tape, inputs). Every tape has at least one node
# that is consumed by more than one later op, to exercise gradient
# accumulation (not overwrite) on reuse.
TAPES = [
    (2, [
        ("mul", (0, 1)),   # v2 = x0*x1
        ("sin", (2,)),     # v3 = sin(v2)
        ("add", (2, 3)),   # v4 = v2 + v3          (v2 reused)
    ], [0.6, 1.1]),
    (3, [
        ("mul", (0, 1)),   # v3 = x0*x1
        ("exp", (2,)),     # v4 = exp(x2)
        ("add", (3, 4)),   # v5 = v3 + v4
        ("mul", (5, 0)),   # v6 = v5 * x0          (x0 reused)
    ], [0.4, -0.7, 0.2]),
    (2, [
        ("sin", (0,)),     # v2 = sin(x0)
        ("mul", (2, 1)),   # v3 = v2 * x1
        ("add", (2, 3)),   # v4 = v2 + v3          (v2 reused)
        ("exp", (4,)),     # v5 = exp(v4)
    ], [0.3, 0.5]),
    (3, [
        ("mul", (0, 0)),   # v3 = x0*x0            (x0 used twice in one op)
        ("mul", (3, 0)),   # v4 = x0^3             (x0 reused again)
        ("add", (1, 2)),   # v5 = x1 + x2
        ("mul", (4, 5)),   # v6 = x0^3 * (x1+x2)
    ], [0.8, -0.3, 0.6]),
]


def grade(sol, fx) -> dict:
    max_err = 0.0
    for n_inputs, tape, inputs in TAPES:
        values = _forward(tape, inputs)
        ref_grad = _numeric_grad(tape, inputs)

        try:
            got = sol.backward_pass(tape, list(values), n_inputs)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}

        if got.shape != (n_inputs,):
            return {"rel_err": 1.0}

        denom = float(np.linalg.norm(ref_grad)) + 1e-12
        err = float(np.linalg.norm(got - ref_grad) / denom)
        max_err = max(max_err, err)

    return {"rel_err": max_err}
