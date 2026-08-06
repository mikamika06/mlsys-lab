def get_valid_backends(config, env):
    """
    Returns a list of valid attention implementation names from:
    'flash_attention_2', 'sdpa', 'eager'.
    Must be ordered from most preferred to least preferred.
    """
    raise NotImplementedError
