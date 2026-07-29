import w


def check(workdir):
    out = {"fullgraph_ok": 0.0, "break_count": -1.0, "matches_eager": 0.0}
    torch = w.torch_or_none()
    if torch is None:
        return w.needs_torch(out)
    import torch._dynamo as dyn

    Model, norm = w.learner_service(workdir)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    m = Model().eval()
    x = norm(torch.randn(4, 64))

    dyn.reset()
    exp = dyn.explain(m, x)
    out["break_count"] = float(exp.graph_break_count)

    dyn.reset()
    try:
        with torch.no_grad():
            got = torch.compile(m, fullgraph=True)(x)
            ref = m(x)
        out["fullgraph_ok"] = 1.0
        out["matches_eager"] = 1.0 if torch.allclose(got, ref, atol=1e-5, rtol=1e-4) else 0.0
        out["max_abs_err"] = float((got - ref).abs().max())
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"fullgraph=True still raises: {type(e).__name__}: {str(e)[:160]}"
    finally:
        dyn.reset()
    return out
