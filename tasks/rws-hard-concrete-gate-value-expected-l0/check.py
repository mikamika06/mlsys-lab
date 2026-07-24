import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _oracle(log_alpha, beta, gamma, zeta):
    gate = np.clip(_sigmoid(log_alpha) * (zeta - gamma) + gamma, 0.0, 1.0)
    l0 = _sigmoid(log_alpha - beta * np.log(-gamma / zeta))
    return gate, l0


def grade(sol, fx) -> dict:
    """
    Builds several seeded random log-alpha vectors and (beta, gamma, zeta)
    settings, computes the deterministic hard-concrete gate value and the
    closed-form expected-L0 probability with a NumPy oracle, and compares
    the submission's two outputs (relative error) to the oracle's.
    """
    rng = np.random.default_rng(0)
    gate_rel_worst = 0.0
    l0_rel_worst = 0.0
    for beta, gamma, zeta in [(2 / 3, -0.1, 1.1), (1.0, -0.2, 1.2), (0.5, -0.05, 1.05)]:
        n = int(rng.integers(8, 24))
        log_alpha = rng.normal(scale=2.0, size=n)

        gate_exp, l0_exp = _oracle(log_alpha, beta, gamma, zeta)

        try:
            gate_got, l0_got = sol.hard_concrete_gate(log_alpha.copy(), beta, gamma, zeta)
            gate_got = np.asarray(gate_got, dtype=np.float64)
            l0_got = np.asarray(l0_got, dtype=np.float64)
        except Exception:
            return {"gate_rel_err": float("inf"), "l0_rel_err": float("inf")}

        if gate_got.shape != gate_exp.shape:
            gate_rel_worst = float("inf")
        else:
            num = np.linalg.norm(gate_got - gate_exp)
            den = np.linalg.norm(gate_exp) + 1e-12
            gate_rel_worst = max(gate_rel_worst, float(num / den))

        if l0_got.shape != l0_exp.shape:
            l0_rel_worst = float("inf")
        else:
            num = np.linalg.norm(l0_got - l0_exp)
            den = np.linalg.norm(l0_exp) + 1e-12
            l0_rel_worst = max(l0_rel_worst, float(num / den))

    return {"gate_rel_err": gate_rel_worst, "l0_rel_err": l0_rel_worst}
