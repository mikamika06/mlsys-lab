import numpy as np
from .offloaded import DynamicCache

def is_legacy_tuple(past_key_values):
    raise NotImplementedError

def port_legacy_to_cache(past_key_values):
    raise NotImplementedError
