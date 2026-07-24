import numpy as np

_EPS = 1e-12


def _gauss(x, mu, sigma):
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return g / g.sum()


def _oracle(p, students):
    p = np.asarray(p, dtype=np.float64)
    Q = np.asarray(students, dtype=np.float64)

    def kl(a, b):
        return np.sum(a * (np.log(a + _EPS) - np.log(b + _EPS)), axis=-1)

    forward = kl(p[None, :], Q)
    reverse = kl(Q, p[None, :])
    m = 0.5 * (p[None, :] + Q)
    jsd = 0.5 * kl(p[None, :], m) + 0.5 * kl(Q, m)

    out = {}
    for name, vals in (("forward_kl", forward), ("reverse_kl", reverse), ("jsd", jsd)):
        vals = np.asarray(vals, dtype=np.float64)
        out[name] = (vals, int(np.argmin(vals)))
    return out


def _make_case(peak1, peak2, w1, sigma, student_sigma, mu_lo, mu_hi, n_mu):
    x = np.linspace(-8.0, 8.0, 161)
    p = w1 * _gauss(x, peak1, sigma) + (1.0 - w1) * _gauss(x, peak2, sigma)
    p = p / p.sum()
    mus = np.linspace(mu_lo, mu_hi, n_mu)
    students = np.stack([_gauss(x, mu, student_sigma) for mu in mus])
    return p, students, mus, peak1, peak2


def _cases():
    return [
        _make_case(-3.0, 3.0, 0.65, 0.8, 1.0, -6.0, 6.0, 49),
        _make_case(-4.0, 2.0, 0.55, 0.6, 0.9, -6.0, 6.0, 61),
    ]


FAIL = {"rel_err": float("inf"), "argmin_exact": 0.0, "pattern_ok": 0.0}


def _classify(mu, peak1, peak2):
    """'covering' if mu sits closer to the midpoint between the two teacher
    modes than to either mode individually; 'seeking' otherwise."""
    midpoint = 0.5 * (peak1 + peak2)
    mid_dist = abs(mu - midpoint)
    peak_dist = min(abs(mu - peak1), abs(mu - peak2))
    return "covering" if mid_dist < peak_dist else "seeking"


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    argmin_exact = 1.0
    pattern_ok = 1.0
    expected_class = {"forward_kl": "covering", "reverse_kl": "seeking", "jsd": "seeking"}

    for p, students, mus, peak1, peak2 in _cases():
        ref = _oracle(p, students)

        try:
            got = sol.kd_divergence_family(np.array(p, copy=True), np.array(students, copy=True))
        except Exception:
            return dict(FAIL)

        if not isinstance(got, dict):
            return dict(FAIL)

        for name in ("forward_kl", "reverse_kl", "jsd"):
            if name not in got:
                return dict(FAIL)
            try:
                vals, amin = got[name]
                vals = np.asarray(vals, dtype=np.float64)
                amin = int(amin)
            except (TypeError, ValueError):
                return dict(FAIL)

            ref_vals, ref_amin = ref[name]
            if vals.shape != ref_vals.shape or not np.all(np.isfinite(vals)):
                return dict(FAIL)

            rel = np.linalg.norm(vals - ref_vals) / (np.linalg.norm(ref_vals) + 1e-12)
            worst_rel = max(worst_rel, float(rel))

            if amin != ref_amin:
                argmin_exact = 0.0

            if not (0 <= amin < len(mus)):
                pattern_ok = 0.0
                continue
            got_class = _classify(float(mus[amin]), peak1, peak2)
            if got_class != expected_class[name]:
                pattern_ok = 0.0

    return {"rel_err": worst_rel, "argmin_exact": argmin_exact, "pattern_ok": pattern_ok}
