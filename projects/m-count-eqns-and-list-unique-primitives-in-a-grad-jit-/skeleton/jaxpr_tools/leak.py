class LeakedTracerError(Exception):
    pass


class Tracer:
    def __init__(self, tracer_id: str, frame_id: int, val: float):
        raise NotImplementedError


class TraceFrame:
    def __init__(self, frame_id: int):
        raise NotImplementedError


class TraceContext:
    @classmethod
    def reset(cls):
        raise NotImplementedError

    @classmethod
    def current_frame(cls):
        raise NotImplementedError

    @classmethod
    def create_tracer(cls, val: float, tracer_id: str = None):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


def detect_leaked_tracers(container):
    raise NotImplementedError


def validate_tracer(tracer):
    raise NotImplementedError


def reproduce_leak(closure_fn, input_val: float = 1.0):
    raise NotImplementedError
