"""IPEX API Classification module."""


def classify_api_call(api_name):
    """Classify an IPEX API call string.

    Returns a dict with keys 'status' ('upstreamed', 'removed', 'retained')
    and 'target' (str or None).
    """
    raise NotImplementedError


def classify_api_batch(api_list):
    """Classify a list of API name strings."""
    raise NotImplementedError
