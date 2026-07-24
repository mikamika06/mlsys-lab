import sys

def set_dict_size_ratio(elements):
    """
    Compute the ratio of the memory footprint of a set to that of a dict
    containing the same elements.
    """
    s = set(elements)
    d = {e: e for e in elements}
    return sys.getsizeof(s) / sys.getsizeof(d)
