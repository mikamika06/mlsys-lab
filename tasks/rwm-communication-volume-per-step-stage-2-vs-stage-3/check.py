def _oracle(phi, stage):
    if stage not in (2, 3):
        raise ValueError("unsupported stage")
    communication_ops = [
        "gradient_reduce_scatter",
        "parameter_all_gather",
    ]
    if stage == 3:
        communication_ops.append("forward_parameter_all_gather")
    volume = 0.0
    for _ in communication_ops:
        volume += phi
    return volume


def grade(sol, fx) -> dict:
    cases = [
        (1.0, 2),
        (7.5, 2),
        (1024.0, 3),
        (1000000.0, 3),
    ]
    ok = 1.0
    for phi, stage in cases:
        try:
            got = sol.compute_comm_volume(phi, stage)
            ref = _oracle(phi, stage)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break

    if ok:
        try:
            sol.compute_comm_volume(10.0, 1)
            ok = 0.0
        except ValueError:
            pass
        except Exception:
            ok = 0.0
    return {"exact_match": ok}
