import gc
import weakref
from typing import List, Tuple

def get_callback_order(objects: List[Tuple[str, object]]) -> List[str]:
    """
    Returns the order in which weak reference callbacks fire.

    Args:
        objects: List of (name, obj) pairs. Each obj is an object that supports
                 weak references (e.g., an instance of a class). The name is a
                 string identifier for recording callback order.

    Returns:
        A list of strings (the names) in the order their weakref callbacks
        were invoked during garbage collection.
    """
    raise NotImplementedError('Implement weakref callback ordering')
