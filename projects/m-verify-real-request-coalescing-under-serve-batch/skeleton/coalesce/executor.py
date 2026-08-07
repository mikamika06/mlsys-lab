import asyncio
import inspect
from typing import Any, Callable, List

class ServeBatchTypeError(TypeError):
    """Raised when @serve.batch decorates a non-async function."""
    pass

def serve_batch(max_batch_size: int = 10, batch_wait_timeout_s: float = 0.05):
    """Decorator mimicking Ray Serve @serve.batch behavior."""
    raise NotImplementedError
