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
    callback_order = []

    # Register weak references with callbacks
    refs = []
    for name, obj in objects:
        # Capture the name in the callback via default argument
        def callback(ref, name=name):
            callback_order.append(name)
        r = weakref.ref(obj, callback)
        refs.append(r)

    # Drop the original objects (the strong references in `objects`)
    # The only remaining references are the weak references in `refs`.
    # Clear the `objects` list to ensure no strong references remain.
    del objects
    del refs  # We don't need the list, but the weak refs are still alive as callbacks hold the name.

    # Force garbage collection to trigger callbacks
    gc.collect()

    return callback_order
