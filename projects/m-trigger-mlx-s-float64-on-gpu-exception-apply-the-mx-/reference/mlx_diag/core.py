import mlx.core as mx


def safe_float64_compute(func, *args, **kwargs):
    try:
        with mx.device(mx.gpu):
            return func(*args, **kwargs)
    except Exception:
        with mx.stream(mx.cpu):
            return func(*args, **kwargs)
