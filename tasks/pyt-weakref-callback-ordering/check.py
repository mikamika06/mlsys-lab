import gc
import weakref
from typing import List, Tuple

def _make_objects(n: int) -> List[Tuple[str, object]]:
    """Create n objects in a known order, each supporting weakrefs."""
    class Obj:
        pass
    # Return objects with names "obj_0", "obj_1", ..., "obj_{n-1}"
    return [(f"obj_{i}", Obj()) for i in range(n)]

def _ref_get_callback_order(objects: List[Tuple[str, object]]) -> List[str]:
    """
    Reference implementation: register weakref callbacks, drop all strong
    references, force collection, and return the order of callback invocation.
    """
    # Keep a list to store callback invocations
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

def grade(sol, fx) -> dict:
    """
    Grade the solution by running it on a few test cases.
    The reference implementation (oracle) is _ref_get_callback_order.
    We compare its output to the student's function's output.
    """
    test_cases = [
        _make_objects(3),
        _make_objects(5),
        _make_objects(1),  # edge case
        _make_objects(0),  # empty list
    ]

    total_pass = 1.0
    for objects in test_cases:
        try:
            # Run the student's function on a COPY of the objects (since the function may modify the list)
            student_objects = list(objects)  # shallow copy
            student_order = sol.get_callback_order(student_objects)
        except Exception as e:
            total_pass = 0.0
            break

        # Run the reference oracle on the ORIGINAL objects (fresh list)
        ref_objects = list(objects)
        ref_order = _ref_get_callback_order(ref_objects)

        if student_order != ref_order:
            total_pass = 0.0
            break

    return {"exact_match": total_pass}
