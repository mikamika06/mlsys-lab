class LeakedTracerError(Exception):
    pass


class Tracer:
    def __init__(self, tracer_id: str, frame_id: int, val: float):
        self.tracer_id = tracer_id
        self.frame_id = frame_id
        self.val = val


class TraceFrame:
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.is_active = True


class TraceContext:
    _active_frames = {}
    _next_id = 1

    @classmethod
    def reset(cls):
        cls._active_frames = {}
        cls._next_id = 1

    @classmethod
    def current_frame(cls):
        if not cls._active_frames:
            return None
        return list(cls._active_frames.values())[-1]

    @classmethod
    def create_tracer(cls, val: float, tracer_id: str = None) -> Tracer:
        frame = cls.current_frame()
        if frame is None:
            raise RuntimeError("No active trace context")
        tid = tracer_id or f"t_{cls._next_id}"
        cls._next_id += 1
        return Tracer(tid, frame.frame_id, val)

    def __enter__(self):
        fid = len(TraceContext._active_frames) + 1
        frame = TraceFrame(fid)
        TraceContext._active_frames[fid] = frame
        self.frame = frame
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.frame.is_active = False


def detect_leaked_tracers(container) -> list:
    leaked = []
    def _search(obj):
        if hasattr(obj, "frame_id") and hasattr(obj, "tracer_id"):
            frame = TraceContext._active_frames.get(obj.frame_id)
            if frame is None or not frame.is_active:
                leaked.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _search(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                _search(item)
    _search(container)
    return leaked


def validate_tracer(tracer: Tracer) -> float:
    frame = TraceContext._active_frames.get(tracer.frame_id)
    if frame is None or not frame.is_active:
        raise LeakedTracerError(f"Tracer {tracer.tracer_id} leaked from inactive frame {tracer.frame_id}")
    return tracer.val


def reproduce_leak(closure_fn, input_val: float = 1.0):
    closure_list = []
    with TraceContext():
        tracer = TraceContext.create_tracer(input_val, tracer_id="tr_closure")
        closure_fn(tracer, closure_list)
    leaked = detect_leaked_tracers(closure_list)
    return closure_list, leaked
