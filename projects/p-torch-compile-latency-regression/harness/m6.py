import w


def check(workdir):
    out = {"speedup": 0.0, "separable": 0.0, "outputs_match": 0.0}
    torch = w.torch_or_none()
    if torch is None:
        return w.needs_torch(out)
    from mlsys.measure import ab

    Base, base_norm = w.baseline()
    Model, norm = w.learner_service(workdir)
    torch.manual_seed(0)
    torch.set_num_threads(4)

    base = Base().eval()
    mine = Model().eval()
    try:
        mine.load_state_dict(base.state_dict())
    except Exception:  # noqa: BLE001
        out["_note"] = "the fixed model no longer accepts the original weights"
        return out

    xs = [torch.randn(b, 64) for b in w.BATCHES]
    compiled = torch.compile(mine, dynamic=True)
    with torch.no_grad():
        worst = 0.0
        for x in xs:
            worst = max(worst, float((compiled(norm(x)) - base(base_norm(x))).abs().max()))
        out["outputs_match"] = 1.0 if worst < 1e-4 else 0.0
        out["max_abs_err"] = worst

        def a():
            for x in xs:
                base(base_norm(x))

        def b():
            for x in xs:
                compiled(norm(x))

        r = ab(a, b, rounds=9)
    out["speedup"] = float(r["ratio"])
    out["separable"] = float(r["separable"])
    out["baseline_ms"] = r["a_median"] * 1000
    out["fixed_ms"] = r["b_median"] * 1000
    if not r["separable"]:
        out["_note"] = ("the two timings overlap: whatever changed is inside the noise, "
                        "so it cannot be claimed as a win")
    return out
