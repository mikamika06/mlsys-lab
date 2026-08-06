import torch


def capture_global_mutation_error(func, *args, **kwargs):
    try:
        torch.export.export(func, args, kwargs)
        return None
    except Exception as e:
        return {"error_type": type(e).__name__, "message": str(e)}
