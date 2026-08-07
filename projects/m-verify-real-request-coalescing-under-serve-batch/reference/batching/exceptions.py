class RayServeSyncException(Exception):
    pass


def handle_sync_function_batch(func, batch):
    if not callable(func):
        raise RayServeSyncException("Function must be callable")
    try:
        return [func(item) for item in batch]
    except Exception as e:
        raise RayServeSyncException(f"Error executing sync function in batch: {e}")
