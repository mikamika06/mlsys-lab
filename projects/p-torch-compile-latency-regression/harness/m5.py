import w


def check(workdir):
    out = {"unique_graphs": -1.0, "outputs_match": 0.0}
    torch = w.torch_or_none()
    if torch is None:
        return w.needs_torch(out)
    import torch._dynamo as dyn

    Model, norm = w.learner_service(workdir)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    m = Model().eval()

    dyn.reset()
    dyn.utils.counters.clear()
    c = torch.compile(m, dynamic=True)
    worst = 0.0
    with torch.no_grad():
        for b in w.BATCHES:
            x = norm(torch.randn(b, 64))
            got, ref = c(x), m(x)
            worst = max(worst, float((got - ref).abs().max()))
    out["unique_graphs"] = float(dyn.utils.counters.get("stats", {}).get("unique_graphs", -1))
    out["outputs_match"] = 1.0 if worst < 1e-4 else 0.0
    out["max_abs_err"] = worst
    dyn.reset()
    return out
