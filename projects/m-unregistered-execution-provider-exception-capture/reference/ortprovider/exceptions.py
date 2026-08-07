"""Exception handling for unregistered execution providers."""

class UnregisteredProviderError(Exception):
    """Raised when an execution provider is not registered or available."""
    pass

def safe_create_session(create_fn, *args, **kwargs):
    try:
        return create_fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "not available" in msg or "unregistered" in msg or "provider" in msg:
            raise UnregisteredProviderError(f"Execution provider unavailable or unregistered: {e}") from e
        raise
