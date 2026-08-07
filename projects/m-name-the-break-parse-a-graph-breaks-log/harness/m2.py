import ref


def check(workdir):
    import torch
    from graphclean.module import CleanModel

    out = {"fullgraph_clean": 0.0}
    try:
        model = CleanModel()
        x = torch.randn(2, 16)

        compiled = torch.compile(model, fullgraph=True, backend="eager")
        _ = compiled(x)
        out["fullgraph_clean"] = 1.0
    except Exception as e:
        out["_note"] = f"fullgraph compilation failed: {type(e).__name__}: {str(e)[:120]}"
    return out
