"""Exception handling for unregistered execution providers."""

class UnregisteredProviderError(Exception):
    pass

def safe_create_session(create_fn, *args, **kwargs):
    raise NotImplementedError
