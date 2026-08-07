class StreamContext:
    def __init__(self, device):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

def get_active_device():
    raise NotImplementedError

def execute_op(op, tensor):
    raise NotImplementedError

def safe_float64_exec(op, tensor):
    raise NotImplementedError
