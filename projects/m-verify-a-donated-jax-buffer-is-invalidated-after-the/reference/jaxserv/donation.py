def is_buffer_deleted(buf) -> bool:
    if hasattr(buf, "is_deleted"):
        if callable(buf.is_deleted):
            return bool(buf.is_deleted())
        return bool(buf.is_deleted)
    if hasattr(buf, "_deleted"):
        return bool(buf._deleted)
    return False


def verify_donation(fn, arg, *extra_args):
    res = fn(arg, *extra_args)
    invalidated = is_buffer_deleted(arg)
    return res, invalidated
