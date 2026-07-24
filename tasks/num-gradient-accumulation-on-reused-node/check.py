import numpy as np

from mlsys import scorers

# tape_A: t = x0*x1 is consumed by THREE downstream ops (sin, mul, final add)
#   node3 = x0*x1
#   node4 = sin(node3)
#   node5 = node3*x2
#   node6 = node4+node5
#   node7 = node6+node3      <- output
_TAPE_A = [
    ("mul", (0, 1)),
    ("sin", (3,)),
    ("mul", (3, 2)),
    ("add", (4, 5)),
    ("add", (6, 3)),
]
_TAPE_A_NIN = 3

# tape_B: s = x0+x1 is consumed by TWO downstream ops (sin, mul) and then the
# mul result AND s itself both feed the final sub.
#   node2 = x0+x1
#   node3 = sin(node2)
#   node4 = node2*node3
#   node5 = node4-node2       <- output
_TAPE_B = [
    ("add", (0, 1)),
    ("sin", (2,)),
    ("mul", (2, 3)),
    ("sub", (4, 2)),
]
_TAPE_B_NIN = 2


def _forward(tape, x):
    """Oracle forward evaluator (real NumPy ops, no autograd)."""
    n_in = x.shape[0]
    val = np.zeros(n_in + len(tape), dtype=np.float64)
    val[:n_in] = x
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
            val[idx] = np.sin(val[a])
        else:
            raise ValueError(f"unknown op {op!r}")
    return val


def _central_fd_grad(tape, x, h=1e-5):
    """Real oracle: central finite differences on the scalar output."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        fp = _forward(tape, xp)[-1]
        fm = _forward(tape, xm)[-1]
        grad[i] = (fp - fm) / (2.0 * h)
    return grad


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ref_all = []
    got_all = []

    for tape, n_in in [(_TAPE_A, _TAPE_A_NIN), (_TAPE_B, _TAPE_B_NIN)]:
        for _ in range(6):
            x = rng.uniform(-2.0, 2.0, size=n_in)
            ref = _central_fd_grad(tape, x)
            try:
                got = sol.tape_grad(tape, x)
                got = np.asarray(got, dtype=np.float64).ravel()
            except Exception:
                return {"rel_err": float("inf")}

            if got.shape != ref.shape or not np.all(np.isfinite(got)):
                return {"rel_err": float("inf")}

            ref_all.append(ref)
            got_all.append(got)

    ref_cat = np.concatenate(ref_all)
    got_cat = np.concatenate(got_all)
    return {"rel_err": scorers.rel_err(ref_cat, got_cat)}
